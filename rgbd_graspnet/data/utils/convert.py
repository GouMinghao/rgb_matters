__author__ = "Minghao Gou"
__version__ = "1.0"

from graspnetAPI import GraspGroup, Grasp, GraspNet
import numpy as np
import os
import cv2
from itertools import product
import copy
import torch
import time


from .transformation import (
    NUM_GRID_X,
    NUM_GRID_Y,
    X_LENGTH,
    Y_LENGTH,
    get_z,
    framexy_depth_2_xyz,
    depth_inpaint,
)
from .gen_label import (
    decode_view_angle_index,
    get_view_angle_index,
)

from .generate_anchor_matrix import (
    generate_views,
    generate_angles,
    viewpoint_params_to_matrix,
)

from .collision import (
    ModelFreeCollisionDetector,
    ModelFreeCollisionDetectorGPU,
    load_cloud,
)
from rgbd_graspnet.constant import GRASPNET_ROOT

X_STEP = X_LENGTH / NUM_GRID_X
Y_STEP = Y_LENGTH / NUM_GRID_Y


def get_workspace_mask(scene_id, camera, ann_id, graspnet_root, dilate_square=2):
    """
    **Input:**

    - scene_id: int of the index of scene.

    - camera: string of camera type.

    - ann_id: int of the annotation index.

    - graspnet_root: string of the root of graspnet.

    - dilate_square: int of dilated square length.

    **Output:**

    - mask: np.array of shape (NUM_GRID_Y, NUM_GRID_X) with dtype=bool of workspace mask.

    """
    mask_path = os.path.join(
        graspnet_root,
        "scenes",
        "scene_%04d" % scene_id,
        camera,
        "label",
        "%04d.png" % ann_id,
    )
    mask = (cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8)
    mask = cv2.resize(mask, (NUM_GRID_X, NUM_GRID_Y))
    mask = cv2.dilate(mask, np.ones((dilate_square, dilate_square))).astype(bool)
    return mask


def convert_grasp(
    label,
    scene_id,
    camera,
    ann_id,
    graspnet_root=GRASPNET_ROOT,
    top_in_grid=10,
    top_in_map=10000,
    top_sample=600,
    topK=200,
    approach_dist=0.05,
    collision_thresh=0.001,
    empty_thresh=0.01,
    nms_t=0.03,
    nms_r=30,
    width_list=[0.1],
    delta_depth_list=[-0.01, 0.0, 0.01],
    delta_translation_array=np.zeros((1, 3), dtype=np.float32),
    flip=False,
    device="cpu",
    inpainting=False,
):
    """
    **Input:**

    - label: numpy array of shape = (NUM_VIEWS * NUM_ANGLES, NUM_GRID_Y, NUM_GRID_X)

    - scene_id: int of the scene id.

    - camera: string of the type of camera, 'realsense' or 'kinect'

    - ann_id: int of the annotation id.

    - dump_path: string of the path to store the image.

    - dump: bool of whether to store the image.

    - show: bool of whether to show the grasp in window.

    **Output:**

    - GraspGroup instance.
    """

    # get into workspace

    label = copy.deepcopy(label).astype(np.float64)
    # print(f'LABEL.shape{label.shape}')
    if flip:
        label = np.flip(label, axis=2)
    workspace_mask = get_workspace_mask(
        scene_id=scene_id,
        camera=camera,
        ann_id=ann_id,
        graspnet_root=graspnet_root,
        dilate_square=2,
    )

    view_angle_num = label.shape[0]
    workspace_mask = np.tile(
        workspace_mask[np.newaxis, :, :], (view_angle_num, 1, 1)
    ).astype(float)
    label = label * workspace_mask

    # gird level max top_in_grid
    grid_arg = np.argsort(label, axis=0)
    remove_mask = grid_arg < (view_angle_num - top_in_grid)
    label[remove_mask] = 0

    f_label = copy.deepcopy(np.reshape(label, (-1)))
    f_label.sort()
    min_thresh = f_label[::-1][top_in_map]
    label[label <= min_thresh] = 0
    origin_gg = GraspGroup()
    graspnet = GraspNet(root=graspnet_root, camera=camera, split="all")

    scene_points, _ = load_cloud(
        scene_idx=scene_id,
        frame_idx=ann_id,
        graspnet_root=graspnet_root,
        camera=camera,
        inpainting=inpainting,
    )
    print("=" * 20)
    tic = time.time()
    if device == "cpu":
        mfcdetector = ModelFreeCollisionDetector(scene_points)
    else:
        tensor_points = torch.tensor(scene_points).to(device)
        mfcdetector = ModelFreeCollisionDetectorGPU(tensor_points)

    _, depth_path, _, _, _, _, _ = graspnet.loadData(scene_id, camera, ann_id)
    depths = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if inpainting:
        depths = depth_inpaint(depths)
    for grid_x in range(NUM_GRID_X):
        for grid_y in range(NUM_GRID_Y):
            label_grid = label[:, grid_y, grid_x]
            view_angle_index = np.where(label_grid)[0]
            view_index_angle_index = decode_view_angle_index(view_angle_index)
            for view_index, angle_index in view_index_angle_index:
                t, R, width, _ = decode_grasp(
                    grid_x=grid_x,
                    grid_y=grid_y,
                    label=label,
                    view_index=view_index,
                    angle_index=angle_index,
                    depths=depths,
                    camera=camera,
                )
                if flip:
                    R[0] = -R[0]
                score = label[
                    get_view_angle_index(view_index, angle_index), grid_y, grid_x
                ]
                origin_gg.add(
                    Grasp(
                        score,
                        width,
                        0.02,
                        0.04,
                        R,
                        t + np.array([0, 0, 0], dtype=np.float64),
                        -1,
                    )
                )

    origin_gg = origin_gg.nms(nms_t, nms_r / 180.0 * np.pi).sort_by_score()
    origin_gg = origin_gg[0 : min(top_sample, len(origin_gg))]

    gg = GraspGroup()
    for grasp in origin_gg:
        for delta_translation in delta_translation_array:
            for width, delta_depth in product(width_list, delta_depth_list):
                # grasp with lower delta_depth(closer to the camera original place) will have higher score whatever the width is.
                # if the delta_depth is the same, grasp with bigger width will have higher score.
                # Smaller delta translation will be better.
                score = (
                    grasp.score
                    + width / 10
                    - delta_depth
                    - np.linalg.norm(delta_translation)
                )
                rotation_matrix = grasp.rotation_matrix
                t = grasp.translation + np.array([0, 0, delta_depth], dtype=np.float64)
                gg.add(
                    Grasp(
                        score,
                        width,
                        0.02,
                        0.04,
                        rotation_matrix,
                        t + delta_translation,
                        -1,
                    )
                )

    if device == "cpu":
        collision_mask, empty_mask = mfcdetector.detect(
            gg,
            approach_dist=approach_dist,
            collision_thresh=collision_thresh,
            return_empty_grasp=True,
            empty_thresh=empty_thresh,
        )
    else:
        collision_mask, empty_mask = mfcdetector.detect(
            torch.tensor(gg.grasp_group_array).to(device),
            approach_dist=approach_dist,
            collision_thresh=collision_thresh,
            return_empty_grasp=True,
            empty_thresh=empty_thresh,
        )

    remain_mask = np.logical_not(np.logical_or(collision_mask, empty_mask))
    gg = GraspGroup(gg.grasp_group_array[remain_mask])
    gg = gg.nms(nms_t, nms_r / 180.0 * np.pi)
    gg.sort_by_score()
    toc = time.time()

    print("Total Grasp:{}, Convert Time Cost:{}".format(len(gg), toc - tic))
    print("=" * 20)
    return gg[: min(len(gg), topK)]


def decode_grasp(grid_x, grid_y, label, view_index, angle_index, depths, camera):
    """
    **Input:**

    - grid_x: int of the x grid.

    - grid_y: int of the y grid.

    - label: numpy array of the label, shape = (7, NUM_GRID_Y, NUM_GRID_X)

    - view_index: int of the index of the view.

    - angle_index: int of the index of the angle.

    **Ouput:**

    - a tuple of (t, R, width, score)

    - t: numpy array of the translation in camera frame, shape = (3, )

    - R: numpy array of the rotation matrix in camera frame, shape = (3, 3)

    - width: float of the width of the gripper. The unit is millimeter.

    - score: float of the score of the grasp.
    """
    pixel_x = round(grid_x * X_STEP)
    pixel_y = round(grid_y * Y_STEP)
    views = generate_views()
    angles = generate_angles()
    view = views[round(view_index)]
    angle = angles[round(angle_index)]
    toward = -view
    R = viewpoint_params_to_matrix(toward, angle)
    z = get_z(depths, grid_x, grid_y, 0)
    x, y, z = framexy_depth_2_xyz(
        pixel_x=pixel_x, pixel_y=pixel_y, depth=z, camera=camera
    )
    t = np.array([x, y, z], dtype=np.float32) / 1000
    score = 1
    width = 0.02
    return t, R, width, score
