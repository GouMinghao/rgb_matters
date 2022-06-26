__author__ = "Minghao Gou"
__version__ = "1.0"

from graspnetAPI import GraspNet
import os
from tqdm import tqdm
import numpy as np

from .generate_anchor_matrix import generate_matrix, NUM_ANGLES, NUM_VIEWS
from .view_rotation import get_towards_and_angles
from .transformation import (
    batch_rgbdxyz_2_rgbxy_depth,
    X_LENGTH,
    Y_LENGTH,
    NUM_GRID_X,
    NUM_GRID_Y,
)


X_STEP = X_LENGTH / NUM_GRID_X
Y_STEP = Y_LENGTH / NUM_GRID_Y


def get_view_angle_index(
    view_index, angle_index, num_views=NUM_VIEWS, num_angles=NUM_ANGLES
):
    """
    **Input:**

    - view_index: int of the index of the view.

    - angle_index: int of the index of the angle.

    - num_views: int of the number of views.

    - num_angles: int of the number of angles.

    **Output:**

    - int of the index of the view-angle combination.
    """
    return view_index * num_angles + angle_index


def decode_view_angle_index(
    view_angle_index, num_views=NUM_VIEWS, num_angles=NUM_ANGLES
):
    """
    **Input:**

    - view_angle_index: int of the index of the view-angle combination.

    - num_views: int of the number of views.

    - num_angles: int of the number of angles.

    **Output:**

    - tuple of the index of view and angle
    """
    view_index = view_angle_index // num_angles
    angle_index = view_angle_index % num_angles
    return np.stack((view_index, angle_index)).T


def get_grasp_label(
    scene_id,
    camera,
    ann_id,
    graspnet,
    grasp_labels=None,
    collision_labels=None,
    grasp_thresh=0.1,
):
    """
    **Input:**

    - scene_id: int of index of scene.

    - camera: string of type of camera, 'realsense' or 'kinect'.

    - ann_id: int of index of annotation.

    - graspnet: GraspNet instance.

    - collision_labels: collision labels read by graspnetAPI.

    - grasp_labels: grasp labels read by graspnetAPI.

    - grasp_thresh: float of grasp coefficient of friction threshold.

    **Output:**

    - numpy array of grasp level label of shape (-1, 8)

    - [grid x, grid y, offset x, offset y, width, z, view index, angle index]
    """
    anchor_matrix = generate_matrix()  # (120, 12, 3, 3)
    # g = GraspNet(root = GRASPNET_ROOT, camera=camera, split='all')
    grasp = graspnet.loadGrasp(
        sceneId=scene_id,
        annId=ann_id,
        format="6d",
        camera=camera,
        grasp_labels=grasp_labels,
        collision_labels=collision_labels,
        fric_coef_thresh=grasp_thresh,
    )
    ################################
    # import pickle
    # f = open('grasp.pkl','rb')
    # grasp = pickle.load(f)

    # widths = np.array([],dtype=np.float32)
    view_indexs = np.array([], dtype=np.float32)
    angle_indexs = np.array([], dtype=np.float32)
    gridxs = np.array([], dtype=np.float32)
    gridys = np.array([], dtype=np.float32)
    # offsetxs = np.array([],dtype=np.float32)
    # offsetys = np.array([],dtype=np.float32)
    # zs = np.array([],dtype=np.float32)
    # for obj_id in tqdm(grasp.keys(),'generating grasp label'):
    # for obj_id in grasp.keys():

    # g_obj = grasp[obj_id]
    # Rs = g_obj["Rs"]
    Rs = grasp.rotation_matrices
    # widths = np.hstack([widths,g_obj['widths']]) # width
    points = grasp.translations
    views_index, angles_index = get_towards_and_angles(Rs, anchor_matrix)
    view_indexs = np.hstack([view_indexs, views_index])  # view index
    angle_indexs = np.hstack([angle_indexs, angles_index])  # angle index
    x, y, z = batch_rgbdxyz_2_rgbxy_depth(points=points, camera=camera)
    gridxs = np.hstack([gridxs, x // X_STEP])
    gridys = np.hstack([gridys, y // Y_STEP])
    # offsetxs = np.hstack([offsetxs, (x % X_STEP) / X_STEP])
    # offsetys = np.hstack([offsetys, (y % Y_STEP) / Y_STEP])
    # zs = np.hstack([zs,z])
    grasp_label = np.stack([gridxs, gridys, view_indexs, angle_indexs]).T  # (-1, 8)
    # filter grasp outside the picture
    mask = (
        (grasp_label[:, 0] >= 0)
        & (grasp_label[:, 0] < NUM_GRID_X)
        & (grasp_label[:, 1] >= 0)
        & (grasp_label[:, 1] < NUM_GRID_Y)
    )
    grasp_label = grasp_label[mask].astype(np.uint32)
    return grasp_label


def get_grid_label(
    scene_id,
    camera,
    ann_id,
    graspnet,
    grasp_labels=None,
    collision_labels=None,
    grasp_thresh=0.1,
):
    """
    **Input:**

    - scene_id: int of index of scene.

    - camera: string of type of camera, 'realsense' or 'kinect'.

    - ann_id: int of index of annotation.

    - graspnet: GraspNet instance.

    - collision_labels: collision labels read by graspnetAPI.

    - grasp_laels: grasp labels read by graspnetAPI.

    - grasp_thresh: float of grasp coefficient of friction threshold.

    **Output:**

    - numpy array of grid level label of shape (7, NUM_GRID_Y, NUM_GRID_X)

    - each grid is an array of [score, offset x, offset y, view, angle, relative_width, delta_z]
    """
    grasp_label = get_grasp_label(
        scene_id=scene_id,
        camera=camera,
        ann_id=ann_id,
        graspnet=graspnet,
        grasp_labels=grasp_labels,
        collision_labels=collision_labels,
        grasp_thresh=grasp_thresh,
    )
    grid_label = np.zeros(
        shape=(NUM_VIEWS * NUM_ANGLES, NUM_GRID_Y, NUM_GRID_X), dtype=bool
    )
    for label in grasp_label:
        gridx, gridy, view_index, angle_index = label.astype(np.uint32)
        grid_label[
            get_view_angle_index(
                view_index=view_index,
                angle_index=angle_index,
                num_views=NUM_VIEWS,
                num_angles=NUM_ANGLES,
            ),
            gridy,
            gridx,
        ] = True  # view * angle

    return grid_label


def batch_get_grid_label(
    scene_id,
    camera,
    ann_id,
    graspnet,
    grasp_labels=None,
    collision_labels=None,
    grasp_thresh=0.1,
):
    """
    **Input:**

    - scene_id: int of index of scene.

    - camera: string of type of camera, 'realsense' or 'kinect'.

    - ann_id: int of index of annotation.

    - graspnet: GraspNet instance.

    - collision_labels: collision labels read by graspnetAPI.

    - grasp_laels: grasp labels read by graspnetAPI.

    - grasp_thresh: float of grasp coefficient of friction threshold.

    **Output:**

    - numpy array of grid level label of shape (7, NUM_GRID_Y, NUM_GRID_X)

    - each grid is an array of [score, offset x, offset y, view, angle, relative_width, delta_z]
    """
    grasp_label = get_grasp_label(
        scene_id=scene_id,
        camera=camera,
        ann_id=ann_id,
        graspnet=graspnet,
        grasp_labels=grasp_labels,
        collision_labels=collision_labels,
        grasp_thresh=grasp_thresh,
    )
    grid_label = np.zeros(
        shape=(NUM_VIEWS * NUM_ANGLES * NUM_GRID_Y * NUM_GRID_X), dtype=bool
    )
    # shape (NUM_VIEWS * NUM_ANGLES )
    gridx = grasp_label[:, 0]
    gridy = grasp_label[:, 1]
    view_index = grasp_label[:, 2]
    angle_index = grasp_label[:, 3]
    print(
        f"shapes:, gridx:{gridx.shape}, gridy:{gridy.shape}, view_index:{view_index.shape}, angle_index:{angle_index.shape}"
    )
    view_angle_index = get_view_angle_index(
        view_index=view_index,
        angle_index=angle_index,
        num_views=NUM_VIEWS,
        num_angles=NUM_ANGLES,
    ).astype(np.uint32)
    print(f"view_angle_index:{view_angle_index}")
    print(f"gridy:{gridy}")
    print(f"gridx:{gridx}")
    index = view_angle_index * NUM_GRID_X * NUM_GRID_Y + gridy * NUM_GRID_X + gridx
    grid_label[index] = True
    grid_label = grid_label.reshape((NUM_VIEWS * NUM_ANGLES, NUM_GRID_Y, NUM_GRID_X))
    return grid_label


def get_label_path(root, scene_id, camera, ann_id):
    """
    **Input:**

    - root: string of the label root.

    - scene_id: int of the index of scene.

    - camera: string of type of camera, 'realsense' or 'kinect'.

    - ann_id: int of the index of annotation.

    **Output:**

    - string of the path of the label npy file.
    """
    return os.path.join(
        root, "scene_%04d" % (scene_id,), camera, "%04d.npy" % (ann_id,)
    )


def gen_camera_label(graspnet_root, scene_id, camera, dump_folder):
    """
    **Input:**

    - graspnet_root: string of the root path for Graspnet dataset.

    - scene_id: int of index of scene.

    - camera: string of the type of camera, 'kinect' or 'realsense'.

    - dump_folder: string of path of dump folder.

    **Output:**

    - No output but dump the grid label to folder.
    """
    assert camera in ["realsense", "kinect"], 'camera should be "realsense" or "kinect'
    g = GraspNet(root=graspnet_root, camera=camera, split="all")
    grasp_labels = g.loadGraspLabels(g.getObjIds(scene_id))
    if not os.path.exists(dump_folder):
        os.mkdir(dump_folder)
    collision_labels = g.loadCollisionLabels(scene_id)
    scene_dir = os.path.join(dump_folder, "scene_%04d" % (scene_id,))
    if not os.path.exists(scene_dir):
        os.mkdir(scene_dir)
    camera_dir = os.path.join(scene_dir, camera)
    if not os.path.exists(camera_dir):
        os.mkdir(camera_dir)
    for ann_id in tqdm(
        range(256), desc=f"scene:{scene_id}, camera={camera}", dynamic_ncols=True
    ):
        grid_label = batch_get_grid_label(
            scene_id=scene_id,
            camera=camera,
            ann_id=ann_id,
            graspnet=g,
            grasp_labels=grasp_labels,
            collision_labels=collision_labels,
        )
        np.save(os.path.join(camera_dir, "%04d.npy" % (ann_id)), grid_label)


def gen_scene_label(graspnet_root, scene_id, dump_folder, camera="both"):
    """
    **Input:**

    - graspnet_root: graspnet dataset root

    - scene_id: int of index of scene.

    - dump_folder: string of path of dump folder.

    - camera: string of which camera to generate.

    **Output:**

    - No output but dump the grid label to folder.
    """
    g = dict()
    cameras = ["realsense", "kinect"] if camera == "both" else [camera]
    g["kinect"] = GraspNet(root=graspnet_root, camera="kinect", split="all")
    g["realsense"] = GraspNet(root=graspnet_root, camera="realsense", split="all")
    grasp_labels = g["kinect"].loadGraspLabels(g["kinect"].getObjIds(scene_id))
    os.makedirs(dump_folder, exist_ok=True)
    collision_labels = g["kinect"].loadCollisionLabels(scene_id)
    scene_dir = os.path.join(dump_folder, "scene_%04d" % (scene_id,))
    os.makedirs(scene_dir, exist_ok=True)
    for camera in cameras:
        camera_dir = os.path.join(scene_dir, camera)
        if not os.path.exists(camera_dir):
            os.mkdir(camera_dir)
        for ann_id in tqdm(
            range(256), desc="scene:{}, camera={}".format(scene_id, camera)
        ):
            grid_label = get_grid_label(
                scene_id=scene_id,
                camera=camera,
                ann_id=ann_id,
                graspnet=g[camera],
                grasp_labels=grasp_labels,
                collision_labels=collision_labels,
            )
            np.save(os.path.join(camera_dir, "%04d.npy" % (ann_id)), grid_label)


def parallel_gen_scene_label(graspnet_root, scene_ids, dump_folder, proc=1):
    if proc == 1:
        for scene_id in scene_ids:
            gen_scene_label(graspnet_root, scene_id, dump_folder)
    else:
        from multiprocessing import Pool

        p = Pool(processes=proc)
        for scene_id in scene_ids:
            p.apply_async(gen_scene_label, args=(graspnet_root, scene_id, dump_folder))
        p.close()
        p.join()


def parallel_gen_camera_label(graspnet_root, scene_ids, camera, dump_folder, proc=1):
    if proc == 1:
        for scene_id in scene_ids:
            gen_camera_label(graspnet_root, scene_id, camera, dump_folder)
    else:
        from multiprocessing import Pool

        p = Pool(processes=proc)
        for scene_id in scene_ids:
            p.apply_async(
                gen_camera_label, args=(graspnet_root, scene_id, camera, dump_folder)
            )
        p.close()
        p.join()
