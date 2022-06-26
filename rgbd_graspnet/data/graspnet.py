__author__ = "mhgou"
__version__ = "1.0"

from graspnetAPI import GraspNet
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms

from .utils.label_loader import LabelLoader
from .utils.normal_loader import NormalLoader
from ..constant import NORMALS_DIR, GRASPNET_ROOT, LABEL_DIR


class GraspNetDataset(Dataset):
    def __init__(
        self,
        graspnet_root=GRASPNET_ROOT,
        label_root=LABEL_DIR,
        normals_dir=NORMALS_DIR,
        use_normal=True,
        camera="kinect",
        split="all",
        grayscale=False,
        colorjitter_scale=0,
        random_crop=0,
        normaljitter_scale=0,
    ):
        """
        Graspnet Dataset.

        Args:
            graspnet_root(str): string of graspnet root path.
            label_root(str): labels dir.
            normals_dir(str): normal dir.
            use_normal(bool): if use normal.
            camera(str): kinect or realsense.
            split(str): graspnet split.
            grayscale(bool): if use gray image.
            color_jitter_scale(float): how much to jitter the image, 0 for no change.
            random_crop(float): size of the cropped image [0-1], 0 for original image.
            normaljitter_scale(float): how much to jitter the normal, 0 for no change.
        """
        self.camera = camera
        self.normals_dir = normals_dir
        self.use_normal = use_normal
        self.graspnet = GraspNet(root=graspnet_root, camera=camera, split=split)
        self.label_loader = LabelLoader(label_root)
        self.normal_loader = NormalLoader(self.normals_dir)
        self.grayscale = grayscale
        self.colorjitter = colorjitter_scale > 0
        self.normaljitter = normaljitter_scale > 0
        resize = transforms.Resize((288, 384))
        colorjitter = transforms.ColorJitter(
            colorjitter_scale, colorjitter_scale, colorjitter_scale, colorjitter_scale
        )
        normaljitter = transforms.ColorJitter(
            normaljitter_scale,
            normaljitter_scale,
            normaljitter_scale,
            normaljitter_scale,
        )
        grayscale = transforms.Grayscale(3)
        totensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        trans_list = [resize]
        if self.colorjitter:
            trans_list.append(colorjitter)
        if self.grayscale:
            trans_list.append(grayscale)
        trans_list += [totensor, normalize]
        self.rgb_transform = transforms.Compose(trans_list)
        normal_trans_list = [resize]
        if self.normaljitter:
            normal_trans_list.append(normaljitter)
        normal_trans_list += [totensor]
        self.normal_transform = transforms.Compose(normal_trans_list)
        self.random_crop = random_crop

    def __len__(self):
        return len(self.graspnet)

    def __getitem__(self, *args):
        """
        Args:
            index(int): int of index or tuple of (scene_id, camera, ann_id)

        Returns:
            np.array: RGB is a numpy array of shape (3, 224, 224).
            np.array: depth is a numpy array of shape (1, 224, 224).
            np.array: Label is a numpy array.
            np.array: Normal is a numpy array of shape (3, 224, 224).
        """
        if isinstance(args[0], int):
            rgb_path, depth_path, _, _, _, scene_name, ann_id = self.graspnet.loadData(
                args[0]
            )
        else:
            rgb_path, depth_path, _, _, _, scene_name, ann_id = self.graspnet.loadData(
                *(args[0])
            )
        scene_id = int(scene_name[-4:])
        if self.use_normal:
            normal = self.normal_loader.load_normal(
                scene_id, self.camera, ann_id, form="pil"
            )

            normal = self.normal_transform(normal)
        else:
            normal = torch.tensor([])
        rgb = Image.open(rgb_path)
        rgb = self.rgb_transform(rgb)

        depth = Image.open(depth_path)
        resize = transforms.Resize((288, 384))
        totensor = transforms.ToTensor()
        depth_trans_list = [resize]
        depth_trans = transforms.Compose(depth_trans_list)
        depth = depth_trans(depth)
        depth = totensor(np.array(depth))

        label = torch.tensor(
            self.label_loader.load_label(
                scene_id=scene_id, camera=self.camera, ann_id=ann_id
            ).astype(np.float32)
        )

        if self.random_crop > 0:
            reduce_size = random.random() * self.random_crop
            new_size = 1 - reduce_size
            origin_shape = np.array(
                [label.shape[1], label.shape[2], rgb.shape[1], rgb.shape[2]],
                dtype=np.int32,
            )
            new_shape = np.round(origin_shape * new_size).astype(np.int32)
            reduced_shape = origin_shape - new_shape
            reduced_height = reduced_shape[[0, 2]]
            reduced_width = reduced_shape[[1, 3]]
            start_height_size = random.random()
            start_width_size = random.random()
            start_height = (reduced_height * start_height_size).astype(np.int32)
            start_width = (reduced_width * start_width_size).astype(np.int32)
            crop_rgb = rgb[
                :,
                start_height[1] : start_height[1] + new_shape[2],
                start_width[1] : start_width[1] + new_shape[3],
            ]
            crop_label = label[
                :,
                start_height[0] : start_height[0] + new_shape[0],
                start_width[0] : start_width[0] + new_shape[1],
            ]
            if self.use_normal:
                crop_normal = normal[
                    :,
                    start_height[1] : start_height[1] + new_shape[2],
                    start_width[1] : start_width[1] + new_shape[3],
                ]
            new_rgb = F.interpolate(
                crop_rgb.unsqueeze(0), (rgb.shape[1], rgb.shape[2])
            )[0]
            new_label = F.interpolate(
                crop_label.unsqueeze(0), (label.shape[1], label.shape[2])
            )[0]
            if self.use_normal:
                new_normal = F.interpolate(
                    crop_normal.unsqueeze(0), (normal.shape[1], normal.shape[2])
                )[0]
            else:
                new_normal = normal
            return new_rgb, depth, new_label, new_normal
        else:
            return rgb, depth, label, normal
