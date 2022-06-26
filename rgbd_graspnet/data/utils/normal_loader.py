__author__ = "Minghao Gou"
__version__ = "1.0"

import numpy as np
import os
from PIL import Image
import cv2


class NormalLoader:
    def __init__(self, normal_root):
        self.normal_root = normal_root

    def load_normal(self, scene_id, camera, ann_id, form="pil"):
        """
        **Input:**

        - scene_id: int of index of scene.

        - camera: string of type of camera, 'realsense' or 'kinect'.

        - ann_id: int of index of annotation.

        - form: string of the return format. 'pil' or 'numpy'.

        **Output:**

        - numpy array or PIL.Image of normal. Range from  0 - 255, dtype np.uint8, order RGB.
        """
        normal_path = self.get_normal_path(scene_id, camera, ann_id)
        normal = cv2.imread(normal_path)  # xyz order
        if form == "numpy":
            return normal
        elif form == "pil":
            return Image.fromarray(normal)
        else:
            raise ValueError(f"Unknown normal format:{form}")

    def get_normal_path(self, scene_id, camera, ann_id):
        """
        **Input:**

        - scene_id: int of index of scene.

        - camera: string of type of camera, 'realsense' or 'kinect'.

        - ann_id: int of index of annotation.

        **Output:**

        - string of normal path.
        """
        return os.path.join(
            self.normal_root, "scene_%04d" % (scene_id,), camera, "%04d.png" % (ann_id,)
        )
