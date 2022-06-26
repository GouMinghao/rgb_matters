__author__ = "Minghao Gou"
__version__ = "1.0"

import cv2
import numpy as np
import open3d as o3d
import os
from graspnetAPI import GraspNet
from tqdm import tqdm
import argparse

from rgbd_graspnet.constant import GRASPNET_ROOT, NORMALS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        default="realsense",
        choices=["realsense", "kinect", "both"],
        help="which camera(s) to generate",
    )
    parser.add_argument("--skip", default=True, help="skip existed normals")
    args = parser.parse_args()
    cameras = ["realsense", "kinect"] if args.camera == "both" else [args.camera]
    for camera in cameras:
        g = GraspNet(GRASPNET_ROOT, camera=camera)
        scene_list = list(range(190))
        for scene_id in scene_list:
            for ann_id in tqdm(
                range(256), "Generating normal for scene {}".format(scene_id)
            ):
                normal_dir = os.path.join(NORMALS_DIR, "scene_%04d" % scene_id, camera)
                normal_path = os.path.join(normal_dir, "%04d.png" % ann_id)
                if args.skip:
                    if os.path.exists(normal_path):
                        continue
                os.makedirs(normal_dir, exist_ok=True)
                pcd = g.loadScenePointCloud(
                    scene_id,
                    camera,
                    ann_id,
                    use_inpainting=True,
                    use_mask=False,
                    use_workspace=False,
                )
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(150))
                normals = np.asarray(pcd.normals)
                normal = normals.reshape((720, 1280, 3))
                normal = ((normal + 1.0) / 2 * 255.0).astype(np.uint8)
                cv2.imwrite(normal_path, normal)
