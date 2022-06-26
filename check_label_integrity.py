__author__ = "Minghao Gou"
__version__ = "1.0"

from tqdm import tqdm
import os
import argparse

from rgbd_graspnet.constant import LABEL_DIR
from rgbd_graspnet.data.utils.gen_label import get_label_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        default="realsense",
        choices=["realsense", "kinect", "both"],
        help="which camera(s) to check",
    )
    args = parser.parse_args()
    cameras = ["realsense", "kinect"] if args.camera == "both" else [args.camera]

    for scene_id in tqdm(range(190), "check scene label integrity"):
        for camera in cameras:
            for ann_id in range(256):
                if not os.path.exists(
                    get_label_path(LABEL_DIR, scene_id, camera, ann_id)
                ):
                    print(
                        "scene:{} camera:{} ann:{} doesn't exists".format(
                            scene_id, camera, ann_id
                        )
                    )
                    break
