__author__ = "Minghao Gou"
__version__ = "1.0"

import argparse
from rgbd_graspnet.data.utils.gen_label import (
    gen_scene_label,
)
from rgbd_graspnet.constant import GRASPNET_ROOT, LABEL_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        default="both",
        choices=["realsense", "kinect", "both"],
        help="which camera(s) to generate",
    )
    args = parser.parse_args()
    for i in range(190):
        gen_scene_label(
            graspnet_root=GRASPNET_ROOT,
            scene_id=i,
            dump_folder=LABEL_DIR,
            camera=args.camera,
        )
