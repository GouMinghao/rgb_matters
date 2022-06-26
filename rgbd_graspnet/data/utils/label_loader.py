__author__ = "Minghao Gou"
__version__ = "1.0"

import numpy as np
import os


class LabelLoader:
    # a class to load label
    def __init__(self, label_root):
        self.label_root = label_root

    def load_label(self, scene_id, camera, ann_id):
        """
        **Input:**

        - scene_id: int of index of scene.

        - camera: string of type of camera, 'realsense' or 'kinect'.

        - ann_id: int of index of annotation.

        **Output:**

        - numpy array of grid level label
        """
        label_path = self.get_label_path(scene_id, camera, ann_id)
        return np.load(label_path)

    def get_label_path(self, scene_id, camera, ann_id):
        """
        **Input:**

        - scene_id: int of index of scene.

        - camera: string of type of camera, 'realsense' or 'kinect'.

        - ann_id: int of index of annotation.

        **Output:**

        - string of label path.
        """
        return os.path.join(
            self.label_root, "scene_%04d" % (scene_id,), camera, "%04d.npy" % (ann_id,)
        )
