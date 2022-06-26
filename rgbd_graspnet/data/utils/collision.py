__author__ = "chenxi-wang, Minghao Gou"
__version__ = "1.0"

import numpy as np
import open3d as o3d
import torch
from PIL import Image
import scipy.io as scio
import os
import cv2

from graspnetAPI import GraspGroup
from graspnetAPI.utils.utils import CameraInfo, create_point_cloud_from_depth_image


class ModelFreeCollisionDetector:
    """Collision detection in scenes without object labels.
    example usage:
        mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
        collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
        collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
        collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                        return_empty_grasp=True, empty_thresh=0.01)
        collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                        return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """

    def __init__(self, scene_points, voxel_size=0.005):
        """Init function. Current finger width and length are fixed.
        Input:
            scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
            voxel_size: [float]
                    used for downsample
        """
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(
        self,
        grasp_group,
        approach_dist=0.03,
        collision_thresh=0.05,
        return_empty_grasp=False,
        empty_thresh=0.01,
        return_ious=False,
        adjust_gripper_centers=False,
    ):
        """Detect collision of grasps.
        Input:
            grasp_group: [GraspGroup, M grasps]
                    the grasps to check
            approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
            collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
            return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
            empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
            return_ious: [bool]
                    if True, return global collision iou and part collision ious
            adjust_gripper_centers: [bool]
                    if True, add an offset to grasp which makes grasp point closer to object center
        Output:
            collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
            [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
            [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
            [optional] grasp_group: [GraspGroup, M grasps]
                    translated grasps
                    only returned when [adjust_gripper_centers] is True

        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:, np.newaxis]
        depths = grasp_group.depths[:, np.newaxis]
        widths = grasp_group.widths[:, np.newaxis]
        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        ## adjust gripper centers
        if adjust_gripper_centers:
            grasp_group, targets = self._adjust_gripper_centers(
                grasp_group, targets, heights, depths, widths
            )

        ## collision detection
        # get collision masks
        (
            left_mask,
            right_mask,
            bottom_mask,
            shifting_mask,
            inner_mask,
        ) = self._get_collision_masks(targets, heights, depths, widths, approach_dist)
        global_mask = left_mask | right_mask | bottom_mask | shifting_mask

        # calculate equivalant volume of each part
        left_right_volume = (
            heights * self.finger_length * self.finger_width / (self.voxel_size**3)
        ).reshape(-1)
        bottom_volume = (
            heights
            * (widths + 2 * self.finger_width)
            * self.finger_width
            / (self.voxel_size**3)
        ).reshape(-1)
        shifting_volume = (
            heights
            * (widths + 2 * self.finger_width)
            * approach_dist
            / (self.voxel_size**3)
        ).reshape(-1)
        volume = left_right_volume * 2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume + 1e-6)

        # get collison mask
        collision_mask = global_iou > collision_thresh

        if not (return_empty_grasp or return_ious or adjust_gripper_centers):
            return collision_mask

        ret_value = [
            collision_mask,
        ]
        if return_empty_grasp:
            inner_volume = (
                heights * self.finger_length * widths / (self.voxel_size**3)
            ).reshape(-1)
            empty_mask = inner_mask.sum(axis=-1) / inner_volume < empty_thresh
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume + 1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume + 1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume + 1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume + 1e-6)
            ret_value.append(
                [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
            )
        if adjust_gripper_centers:
            ret_value.append(grasp_group)
        return ret_value

    def _adjust_gripper_centers(self, grasp_group, targets, heights, depths, widths):
        ## get point masks
        # height mask
        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        # left finger mask
        mask2 = (targets[:, :, 0] > depths - self.finger_length) & (
            targets[:, :, 0] < depths
        )
        mask4 = targets[:, :, 1] < -widths / 2
        # right finger mask
        mask6 = targets[:, :, 1] > widths / 2
        # get inner mask of each point
        inner_mask = mask1 & mask2 & (~mask4) & (~mask6)

        ## adjust targets and gripper centers
        # get point bounds
        targets_y = targets[:, :, 1].copy()
        targets_y[~inner_mask] = 0
        ymin = targets_y.min(axis=1)
        ymax = targets_y.max(axis=1)
        # get offsets
        offsets = np.zeros([targets.shape[0], np.newaxis, 3], dtype=targets.dtype)
        offsets[:, 1] = (ymin + ymax) / 2
        # adjust targets
        targets[:, :, 1] -= offsets[:, np.newaxis, 1]
        # adjust gripper centers
        R = grasp_group.rotation_matrices
        grasp_group.translations += np.matmul(R, offsets[:, :, np.newaxis]).squeeze(2)

        return grasp_group, targets

    def _get_collision_masks(self, targets, heights, depths, widths, approach_dist):
        # height mask
        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        # left finger mask
        mask2 = (targets[:, :, 0] > depths - self.finger_length) & (
            targets[:, :, 0] < depths
        )
        mask3 = targets[:, :, 1] > -(widths / 2 + self.finger_width)
        mask4 = targets[:, :, 1] < -widths / 2
        # right finger mask
        mask5 = targets[:, :, 1] < (widths / 2 + self.finger_width)
        mask6 = targets[:, :, 1] > widths / 2
        # bottom mask
        mask7 = (targets[:, :, 0] <= depths - self.finger_length) & (
            targets[:, :, 0] > depths - self.finger_length - self.finger_width
        )
        # shifting mask
        mask8 = (
            targets[:, :, 0] <= depths - self.finger_length - self.finger_width
        ) & (
            targets[:, :, 0]
            > depths - self.finger_length - self.finger_width - approach_dist
        )

        # get collision mask of each point
        left_mask = mask1 & mask2 & mask3 & mask4
        right_mask = mask1 & mask2 & mask5 & mask6
        bottom_mask = mask1 & mask3 & mask5 & mask7
        shifting_mask = mask1 & mask3 & mask5 & mask8
        inner_mask = mask1 & mask2 & (~mask4) & (~mask6)

        return left_mask, right_mask, bottom_mask, shifting_mask, inner_mask

    def _crop_scenes_in_sphere(self, points, radius):
        """Crop batch scenes in a sphere space. The scenes need to be centralized.
        Note that the returned scenes are not strictly in a box, but contain all the
            points in the box and have the same point number.
        Input:
            points: [numpy.ndarray, [M,N,3], numpy.float32]
                    batch centralized scene points, batch_size=M, num_points=N
            radius: [float]
                    cropping size
        Output:
            points: [numpy.ndarray, [M,N',3](N'<=N), numpy.float32]
                    batch cropped scenes containing all points in the box (and several outer points)
        """
        dists = np.linalg.norm(points, axis=-1)
        inner_mask = dists < radius
        num_samples = np.sum(inner_mask, axis=-1).max()
        point_indices = np.argsort(dists, axis=-1)
        point_indices = np.tile(point_indices[:, :num_samples, np.newaxis], [1, 1, 3])
        points = np.take_along_axis(points, point_indices, axis=1)
        return points


class ModelFreeCollisionDetectorGPU:
    """Collision detection in scenes without object labels, GPU mode.
    Note that when pre_voxeled is set to True, the scene points should be voxeled before detection and voxel_size is only used for iou computation, which is different from CPU mode.
    example usage:
        mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
        collision_mask = mfcdetector.detect(grasp_group_array, approach_dist=0.03)
        collision_mask, iou_list = mfcdetector.detect(grasp_group_array, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
        collision_mask, empty_mask = mfcdetector.detect(grasp_group_array, approach_dist=0.03, collision_thresh=0.05,
                                        return_empty_grasp=True, empty_thresh=0.01)
        collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group_array, approach_dist=0.03, collision_thresh=0.05,
                                        return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """

    def __init__(
        self, scene_points, voxel_size=0.005, pre_voxeled=False, use_robotiq=False
    ):
        """Init function. Current finger width and length are fixed.
        Input:
            scene_points: [torch.Tensor, (N,3), torch.float32]
                    the scene points to detect
            voxel_size: [float]
                    used for downsampling(when pre_voxeled=True) & iou computation
            pre_voxeled: [bool]
                    ignore downsampling step when set to True, only for point clouds already voxelized
            use_robotiq: [bool]
                    use robotiq gripper shape for collision detection
        """
        self.use_robotiq = use_robotiq
        self.finger_width = 0.015
        if self.use_robotiq:
            self.finger_length = 0.04
            self.bottom_outlier = 0.02
            self.bottom_length = 0.04
        else:
            self.finger_length = 0.06
        self.voxel_size = voxel_size
        self.device = scene_points.device

        if not pre_voxeled:
            scene_cloud = o3d.geometry.PointCloud()
            scene_cloud.points = o3d.utility.Vector3dVector(scene_points.cpu().numpy())
            scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
            scene_points = np.array(scene_cloud.points, dtype=np.float32)
            scene_points = torch.from_numpy(scene_points).to(self.device)
        self.scene_points = scene_points

    def detect(
        self,
        grasp_group_array,
        approach_dist=0.03,
        collision_thresh=0.05,
        return_empty_grasp=False,
        empty_thresh=0.01,
        return_ious=False,
        adjust_gripper_centers=False,
    ):
        """Detect collision of grasps.
        Input:
            grasp_group_array: [torch.Tensor, (M,17), torch.float32]
                    the grasps to check
            approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
            collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
            return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
            empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
            return_ious: [bool]
                    if True, return global collision iou and part collision ious
            adjust_gripper_centers: [bool]
                    if True, add an offset to grasp which makes grasp point closer to object center
        Output:
            collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
            [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
            [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
            [optional] grasp_group: [GraspGroup, M grasps]
                    translated grasps
                    only returned when [adjust_gripper_centers] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group_array[:, 13:16]
        R = grasp_group_array[:, 4:13].view(-1, 3, 3)
        heights = grasp_group_array[:, 2:3]
        depths = grasp_group_array[:, 3:4]
        widths = grasp_group_array[:, 1:2]
        targets = self.scene_points.unsqueeze(0) - T.unsqueeze(1)
        targets = torch.matmul(targets, R)

        ## adjust gripper centers
        if adjust_gripper_centers:
            grasp_group_array, targets = self._adjust_gripper_centers(
                grasp_group_array, targets, heights, depths, widths
            )

        # use adjusted gripper centers and widths
        T = grasp_group_array[:, 13:16]
        widths = grasp_group_array[:, 1:2]

        ## collision detection
        # get collision masks
        (
            left_mask,
            right_mask,
            bottom_mask,
            shifting_mask,
            inner_mask,
        ) = self._get_collision_masks(targets, heights, depths, widths, approach_dist)
        global_mask = left_mask | right_mask | bottom_mask | shifting_mask

        # calculate equivalant volume of each part
        left_right_volume = (
            heights * self.finger_length * self.finger_width / (self.voxel_size**3)
        ).reshape(-1)
        bottom_volume = (
            heights
            * (widths + 2 * self.finger_width)
            * self.finger_width
            / (self.voxel_size**3)
        ).reshape(-1)
        shifting_volume = (
            heights
            * (widths + 2 * self.finger_width)
            * approach_dist
            / (self.voxel_size**3)
        ).reshape(-1)
        volume = left_right_volume * 2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = (global_mask.sum(axis=1) / (volume + 1e-6)).detach().cpu().numpy()

        # get collison mask
        collision_mask = global_iou > collision_thresh

        if not (return_empty_grasp or return_ious or adjust_gripper_centers):
            return collision_mask

        ret_value = [
            collision_mask,
        ]
        if return_empty_grasp:
            inner_volume = (
                heights * self.finger_length * widths / (self.voxel_size**3)
            ).reshape(-1)
            empty_mask = inner_mask.sum(axis=-1) / inner_volume < empty_thresh
            ret_value.append(empty_mask.detach().cpu().numpy())
        if return_ious:
            left_iou = (
                (left_mask.sum(axis=1) / (left_right_volume + 1e-6))
                .detach()
                .cpu()
                .numpy()
            )
            right_iou = (
                (right_mask.sum(axis=1) / (left_right_volume + 1e-6))
                .detach()
                .cpu()
                .numpy()
            )
            bottom_iou = (
                (bottom_mask.sum(axis=1) / (bottom_volume + 1e-6))
                .detach()
                .cpu()
                .numpy()
            )
            shifting_iou = (
                (shifting_mask.sum(axis=1) / (shifting_volume + 1e-6))
                .detach()
                .cpu()
                .numpy()
            )
            ret_value.append(
                [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
            )
        if adjust_gripper_centers:
            ret_value.append(GraspGroup(grasp_group_array.detach().cpu().numpy()))
        return ret_value

    def _adjust_gripper_centers(
        self, grasp_group_array, targets, heights, depths, widths
    ):
        ## get point masks
        # height mask
        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        # left finger mask
        mask2 = (targets[:, :, 0] > depths - self.finger_length) & (
            targets[:, :, 0] < depths
        )
        mask3 = targets[:, :, 1] >= -widths / 2
        # right finger mask
        mask4 = targets[:, :, 1] <= widths / 2
        # get inner mask of each point
        inner_mask = mask1 & mask2 & mask3 & mask4

        ## adjust targets and gripper centers
        # get point bounds
        targets_y = targets[:, :, 1].clone()
        targets_y[~inner_mask] = 0
        ymin, _ = targets_y.min(dim=1)
        ymax, _ = targets_y.max(dim=1)

        # get offsets
        offsets = torch.zeros(
            targets.shape[0], 3, dtype=targets.dtype, device=self.device
        )
        offsets[:, 1] = (ymin + ymax) / 2
        # adjust targets
        targets[:, :, 1] -= offsets[:, 1:2]
        # adjust gripper centers
        R = grasp_group_array[:, 4:13].view(-1, 3, 3)
        grasp_group_array[:, 13:16] += torch.matmul(R, offsets.unsqueeze(2)).squeeze(2)
        grasp_group_array[:, 1] = 1.7 * (ymax - ymin)

        return grasp_group_array, targets

    def _get_collision_masks(self, targets, heights, depths, widths, approach_dist):
        # height mask
        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        # left finger mask
        mask2 = (targets[:, :, 0] > depths - self.finger_length) & (
            targets[:, :, 0] < depths
        )
        mask3 = targets[:, :, 1] > -(widths / 2 + self.finger_width)
        mask4 = targets[:, :, 1] < -widths / 2
        # right finger mask
        mask5 = targets[:, :, 1] < (widths / 2 + self.finger_width)
        mask6 = targets[:, :, 1] > widths / 2
        if self.use_robotiq:
            # bottom mask
            mask7 = (targets[:, :, 0] <= depths - self.finger_length) & (
                targets[:, :, 0] > depths - self.finger_length - self.bottom_length
            )
            mask8 = (
                targets[:, :, 1]
                >= -(widths / 2 + self.finger_width + self.bottom_outlier)
            ) & (
                targets[:, :, 1]
                <= (widths / 2 + self.finger_width + self.bottom_outlier)
            )
            # shifting mask
            mask9 = (
                targets[:, :, 0] <= depths - self.finger_length - self.bottom_length
            ) & (
                targets[:, :, 0]
                > depths - self.finger_length - self.bottom_length - approach_dist
            )
        else:
            # bottom mask
            mask7 = (targets[:, :, 0] <= depths - self.finger_length) & (
                targets[:, :, 0] > depths - self.finger_length - self.finger_width
            )
            # shifting mask
            mask9 = (
                targets[:, :, 0] <= depths - self.finger_length - self.finger_width
            ) & (
                targets[:, :, 0]
                > depths - self.finger_length - self.finger_width - approach_dist
            )

        # get collision mask of each point
        left_mask = mask1 & mask2 & mask3 & mask4
        right_mask = mask1 & mask2 & mask5 & mask6
        if self.use_robotiq:
            bottom_mask = mask1 & mask7 & mask8
            shifting_mask = mask1 & mask8 & mask9
        else:
            bottom_mask = mask1 & mask3 & mask5 & mask7
            shifting_mask = mask1 & mask3 & mask5 & mask9
        inner_mask = mask1 & mask2 & (~mask4) & (~mask6)

        return left_mask, right_mask, bottom_mask, shifting_mask, inner_mask


def transform_points(points, trans):
    ones = np.ones([points.shape[0], 1], dtype=points.dtype)
    # print(2)
    points_ = np.concatenate([points, ones], axis=-1)
    # print(points_.shape)
    # print(trans.shape)
    points_ = np.matmul(trans, points_.T).T
    # print(4)
    return points_[:, :3]


def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
    if trans is not None:
        cloud = transform_points(cloud, trans)

    foreground = cloud[seg > 0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = (cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier)
    mask_y = (cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier)
    mask_z = (cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier)
    workspace_mask = mask_x & mask_y & mask_z
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])
    return workspace_mask


def load_cloud(
    scene_idx,
    frame_idx,
    graspnet_root,
    camera="kinect",
    remove_outlier=True,
    inpainting=False,
):
    # load data
    scene_path = os.path.join(graspnet_root, "scenes", "scene_%04d" % scene_idx, camera)
    color = (
        np.array(Image.open(os.path.join(scene_path, "rgb", "%04d.png" % frame_idx)))
        / 255.0
    )
    depth = np.array(
        Image.open(os.path.join(scene_path, "depth", "%04d.png" % frame_idx))
    ).astype(np.float32)
    seg = np.array(
        Image.open(os.path.join(scene_path, "label", "%04d.png" % frame_idx))
    )
    meta = scio.loadmat(os.path.join(scene_path, "meta", "%04d.mat" % frame_idx))
    # parse metadata
    intrinsic = meta["intrinsic_matrix"]
    factor_depth = meta["factor_depth"]
    camerainfo = CameraInfo(
        1280.0,
        720.0,
        intrinsic[0][0],
        intrinsic[1][1],
        intrinsic[0][2],
        intrinsic[1][2],
        factor_depth,
    )

    if inpainting:
        fault_mask = depth < 200
        depth[fault_mask] = 0
        inpainting_mask = (np.abs(depth) < 10).astype(np.uint8)
        # print(depth.dtype)
        # print(inpainting_mask.dtype)
        depth = cv2.inpaint(depth, inpainting_mask, 5, cv2.INPAINT_NS)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camerainfo, organized=True)
    # get valid points
    depth_mask = depth > 0
    if remove_outlier:
        camera_poses = np.load(os.path.join(scene_path, "camera_poses.npy"))
        align_mat = np.load(os.path.join(scene_path, "cam0_wrt_table.npy"))
        trans = np.dot(align_mat, camera_poses[frame_idx])
        workspace_mask = get_workspace_mask(
            cloud, seg, trans=trans, organized=True, outlier=0.02
        )
        mask = depth_mask & workspace_mask
    else:
        mask = depth_mask
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    return cloud_masked, color_masked
