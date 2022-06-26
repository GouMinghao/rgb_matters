__author__ = "Minghao Gou"
__version__ = "1.0"

import numpy as np
import open3d as o3d
import argparse
import torch
import time
import matplotlib.pyplot as plt

from rgbd_graspnet.data.utils.collision import load_cloud

from rgbd_graspnet.data import GraspNetDataset
from rgbd_graspnet.net.rgb_normal_net import RGBNormalNet
from rgbd_graspnet.data.utils.convert import convert_grasp, get_workspace_mask
from rgbd_graspnet.constant import GRASPNET_ROOT, LABEL_DIR


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--cuda", default=True, type=str2bool, help="Use CUDA to train model"
)
parser.add_argument(
    "--dataset_root", default=GRASPNET_ROOT, help="Dataset root directory path"
)
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    help="Checkpoint state_dict file to resume training from",
)
parser.add_argument("--scene_id", default=18, type=int, help="Scene index")
parser.add_argument("--ann_id", default=10, type=int, help="Annotation index")
parser.add_argument("--camera", default="realsense", help="Camera type")
parser.add_argument(
    "--num_layers", default=50, type=int, help="Number of layers of resnet."
)
parser.add_argument(
    "--kinect_label_root", default=LABEL_DIR, help="Root folder of kinect labels"
)
parser.add_argument(
    "--realsense_label_root",
    default=LABEL_DIR,
    help="Root folder of realsense labels",
)
parser.add_argument("--use_normal", type=str2bool, default=False)
parser.add_argument("--normal_only", type=str2bool, default=False)
args = parser.parse_args()

print(args)

weights_path = args.resume
device = "cuda:0" if args.cuda else "cpu"

net = RGBNormalNet(
    num_layers=args.num_layers, use_normal=args.use_normal, normal_only=args.normal_only
)
state_dict = torch.load(weights_path)
net.load_state_dict(state_dict["net"], strict=False)
net = net.to(device)
net.eval()
print("network loaded")

if args.camera == "kinect":
    test_label_root = args.kinect_label_root
else:
    test_label_root = args.realsense_label_root

graspnet_dataset = GraspNetDataset(
    graspnet_root=args.dataset_root,
    use_normal=args.use_normal,
    label_root=test_label_root,
    camera=args.camera,
    split="all",
    grayscale=False,
    colorjitter_scale=0,
    random_crop=0,
)

rgb, _, label, normal = graspnet_dataset[args.scene_id, args.camera, args.ann_id]

rgb = rgb.unsqueeze(0).to(device)
normal = normal.unsqueeze(0).to(device)
label = label.unsqueeze(0).to(device)

# the first time it will run very slowly.
prob_map = net(rgb, normal)

tic = time.time()
for _ in range(100):
    prob_map = net(rgb, normal)
toc = time.time()

print("=" * 20)
print("Net time:{}".format((toc - tic) / 100.0))
print("=" * 20)

pred_map = prob_map[0].to("cpu").clone().detach().numpy().astype(np.float32)
origin_label = label.to("cpu").clone().detach().numpy()[0]

gg = convert_grasp(
    label=pred_map,
    scene_id=args.scene_id,
    camera=args.camera,
    ann_id=args.ann_id,
    graspnet_root=args.dataset_root,
    top_in_grid=5,
    top_in_map=1000,
    top_sample=200,
    topK=30,
    approach_dist=0.05,
    collision_thresh=0.001,
    empty_thresh=0.10,
    nms_t=0.04,
    nms_r=30,
    width_list=[0.1],
    delta_depth_list=[-0.02, 0, 0.02],
    flip=False,
    device="cuda:0",
)

gg.sort_by_score()

scene_points, colors = load_cloud(
    scene_idx=args.scene_id,
    frame_idx=args.ann_id,
    graspnet_root=GRASPNET_ROOT,
    camera=args.camera,
)
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(scene_points)
pc.colors = o3d.utility.Vector3dVector(colors)

gg_3d = gg.to_open3d_geometry_list()

o3d.visualization.draw_geometries([pc, *gg_3d])

workspace_mask = get_workspace_mask(
    args.scene_id, args.camera, args.ann_id, args.dataset_root
)

plt.subplot(2, 2, 1)
plt.title("Transformed image")
rgbbp = rgb.detach().cpu().numpy()[0]
rgbbp = rgbbp / rgbbp.max()
plt.imshow(np.transpose(rgbbp, (1, 2, 0)))

plt.subplot(2, 2, 2)
plt.title("Workspace mask")
plt.imshow(workspace_mask.astype(float))

plt.subplot(2, 2, 3)
plt.title("Predicted sum of AVH")
pred_heatmap = np.sum(pred_map, axis=0)
pred_heatmap = pred_heatmap / pred_heatmap.max()
plt.imshow(pred_heatmap)

plt.subplot(2, 2, 4)
plt.title("Ground truth sum of AVH")
label_heatmap = np.sum(origin_label, axis=0)
label_heatmap = label_heatmap / label_heatmap.max()
plt.imshow(label_heatmap)

plt.show()
