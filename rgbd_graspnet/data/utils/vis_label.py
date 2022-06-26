__author__ = "Minghao Gou"
__version__ = "1.0"

import open3d as o3d
import numpy as np
from graspnetAPI import GraspNet
import cv2
import os
from PIL import Image


from .gen_label import decode_view_angle_index
from .transformation import get_camera_intrinsic
from .generate_anchor_matrix import (
    generate_views,
    generate_angles,
    viewpoint_params_to_matrix,
)
from .transformation import (
    NUM_GRID_X,
    NUM_GRID_Y,
    X_LENGTH,
    Y_LENGTH,
    get_z,
    framexy_depth_2_xyz,
)

X_STEP = X_LENGTH / NUM_GRID_X
Y_STEP = Y_LENGTH / NUM_GRID_Y


class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def generate_scene_pointcloud(dataset_root, scene_name, anno_idx, camera="kinect"):
    colors = (
        np.array(
            Image.open(
                os.path.join(
                    dataset_root,
                    "scenes",
                    scene_name,
                    camera,
                    "rgb",
                    "%04d.png" % anno_idx,
                )
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    depths = np.array(
        Image.open(
            os.path.join(
                dataset_root,
                "scenes",
                scene_name,
                camera,
                "depth",
                "%04d.png" % anno_idx,
            )
        )
    )
    intrinsics = get_camera_intrinsic(camera)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    s = 1000.0

    xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = points_z > 0
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask]
    colors = colors[mask]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    box = o3d.geometry.TriangleMesh()
    vertices = np.array(
        [
            [0, 0, 0],
            [width, 0, 0],
            [0, 0, depth],
            [width, 0, depth],
            [0, height, 0],
            [width, height, 0],
            [0, height, depth],
            [width, height, depth],
        ]
    )
    vertices[:, 0] += dx
    vertices[:, 1] += dy
    vertices[:, 2] += dz
    triangles = np.array(
        [
            [4, 7, 5],
            [4, 6, 7],
            [0, 2, 4],
            [2, 6, 4],
            [0, 1, 2],
            [1, 3, 2],
            [1, 5, 7],
            [1, 7, 3],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 1],
            [1, 4, 5],
        ]
    )
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box


def plot_gripper_pro_max(center, R, width, depth, score=1):
    """
    center: target point
    R: rotation matrix
    """
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    color_r = score  # red for high score
    color_b = 1 - score  # blue for low score
    color_g = 0
    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate(
        [left_points, right_points, bottom_points, tail_points], axis=0
    )
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate(
        [left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0
    )
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper


def vis_grasp(
    label,
    scene_id,
    camera,
    ann_id,
    graspnet_root,
    dump_path=None,
    dump=False,
    show=True,
    num_grasp=100,
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

    No output but store or(and) show the rendered image.
    """
    if (dump_path is None) and (dump):
        raise ValueError("You need to specify the dump path to store the image.")
    graspnet = GraspNet(root=graspnet_root, camera=camera, split="all")
    _, depth_path, _, _, _, _, _ = graspnet.loadData(scene_id, camera, ann_id)
    depths = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    grasp_geometry = []
    scene_pcd = generate_scene_pointcloud(
        graspnet_root, "scene_%04d" % (scene_id,), ann_id, camera
    )
    cnt = 0
    show_rate = num_grasp / np.sum(label)
    for grid_x in range(NUM_GRID_X):
        for grid_y in range(NUM_GRID_Y):
            label_grid = label[:, grid_y, grid_x]
            view_angle_index = np.where(label_grid > 0.3)[0]
            view_index_angle_index = decode_view_angle_index(view_angle_index)

            for view_index, angle_index in view_index_angle_index:
                # for angle_index in range(NUM_ANGLES):
                if np.random.random() > 1 - show_rate:  # to avoid too many grasp
                    t, R, width, score = decode_grasp(
                        grid_x=grid_x,
                        grid_y=grid_y,
                        label=label,
                        view_index=view_index,
                        angle_index=angle_index,
                        depths=depths,
                        camera=camera,
                    )
                    # The unit for plot_gripper_pro_max is meter rather than millimeter

                    cnt += 1
                    grasp_geometry.append(
                        plot_gripper_pro_max(
                            center=t / 1000,
                            R=R,
                            width=width / 1000,
                            depth=0.015,
                            score=score,
                        )
                    )
                    if cnt == 1000:
                        if show:
                            o3d.visualization.draw_geometries(
                                [*grasp_geometry, scene_pcd]
                            )
    print("Total Grasp:{}".format(cnt))

    if show:
        o3d.visualization.draw_geometries([*grasp_geometry, scene_pcd])


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
    # in picture, the label should be label[grid_y,grid_x]
    pixel_x = round(grid_x * X_STEP)
    pixel_y = round(grid_y * Y_STEP)
    views = generate_views()
    angles = generate_angles()
    view = views[round(view_index)]
    angle = angles[round(angle_index)]
    toward = -view
    R = viewpoint_params_to_matrix(toward, angle)
    z = get_z(depths, grid_x, grid_y, 1)
    x, y, z = framexy_depth_2_xyz(
        pixel_x=pixel_x, pixel_y=pixel_y, depth=z, camera=camera
    )
    t = np.array([x, y, z], dtype=np.float32)
    score = 1
    width = 20
    return t, R, width, score
