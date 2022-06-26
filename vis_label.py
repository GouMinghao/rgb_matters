__author__ = "Minghao Gou"
__version__ = "1.0"

import argparse

from rgbd_graspnet.data.utils.label_loader import LabelLoader
from rgbd_graspnet.data.utils.vis_label import vis_grasp
from rgbd_graspnet.constant import GRASPNET_ROOT, LABEL_DIR


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root", default=GRASPNET_ROOT, help="Dataset root directory path"
    )
    parser.add_argument("--label_root", default=LABEL_DIR, help="Root folder of labels")
    parser.add_argument("--scene_id", default=0, type=int, help="Scene index")
    parser.add_argument("--ann_id", default=0, type=int, help="Annotation index")
    parser.add_argument("--camera", default="realsense", help="Camera type")

    args = parser.parse_args()
    label_loader = LabelLoader(args.label_root)
    label = label_loader.load_label(
        scene_id=args.scene_id, camera=args.camera, ann_id=args.ann_id
    )

    vis_grasp(
        label,
        args.scene_id,
        args.camera,
        args.ann_id,
        show=True,
        graspnet_root=args.dataset_root,
    )
