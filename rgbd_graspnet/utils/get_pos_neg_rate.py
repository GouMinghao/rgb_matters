__author__ = "Minghao Gou"
__version__ = "1.0"

import numpy as np
from tqdm import tqdm

from ..data.utils.label_loader import LabelLoader


def get_camera_pos_neg_number(label_root, scene_id, camera):
    label_loader = LabelLoader(label_root)
    pos = 0
    neg = 0
    for ann_id in range(256):
        label = label_loader.load_label(scene_id=scene_id, camera=camera, ann_id=ann_id)
        total = label.size
        pos_ann = np.sum(label)
        neg_ann = total - pos_ann
        pos += pos_ann
        neg += neg_ann
        print(f"ann_id:{ann_id}, pos:{pos_ann}, neg:{neg_ann}, total:{total}")
    return pos, neg


def get_train_pos_neg_number(label_root, camera):
    label_loader = LabelLoader(label_root)
    pos = 0
    neg = 0
    for scene_id in tqdm(range(100), "calculate scenes"):
        for ann_id in range(256):
            label = label_loader.load_label(
                scene_id=scene_id, camera=camera, ann_id=ann_id
            )
            total = label.size
            pos_ann = np.sum(label)
            neg_ann = total - pos_ann
            pos += pos_ann
            neg += neg_ann
    return pos, neg
