__author__ = "Minghao Gou"
__version__ = "1.0"

import numpy as np
import open3d as o3d

NUM_VIEWS = 60
NUM_ANGLES = 6


def generate_angles(N=NUM_ANGLES):
    """
    **Input:**

    - N: int of number of angles

    **Output:**

    - numpy array of shape (N,), range from 0 to pi.
    """
    return np.pi * np.arange(N, dtype=np.float32) / N


def generate_views(
    N=NUM_VIEWS, phi=(np.sqrt(5) - 1) / 2, center=np.zeros(3, dtype=np.float32), R=1
):
    """
    **Input:**

    - N: int of number of angles IN THE UPPER SEMISPHERE.

    - center: numpy array of shape (3,) of the center coordinates.

    - R: float of radius

    **Output:**

    - numpy array of shape (N,3) of the generated views coordinates.
    """
    idxs = np.arange(
        2 * N, dtype=np.float32
    )  # 2N because half of the views are in the lower semisphere.
    Z = (2 * idxs + 1) / (2 * N) - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X, Y, Z], axis=1)
    views = R * np.array(views) + center
    # toward = - view, so that view[:,2] should be < 0 as the toward[:,2] should be > 0 in CAMERA FRAME.
    mask = views[:, 2] < 0
    assert np.sum(mask) == N
    return views[mask]


def viewpoint_params_to_matrix(towards, angle):
    """
    **Input:**

    - towards: numpy array of shape (3,) of the toward vector

    - angle: float of the inplane rotation angle

    **Output:**

    - numpy array of shape (3,3) of the output rotation matrix
    """
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix


def cross_viewpoint_params_to_matrix(batch_towards, batch_angles):
    """
    **Input:**

    - batch_towards: numpy array of shape (num_views, 3) of the toward vectors.

    - batch_angles: numpy array of shape (num_angles,) of the inplane rotation angles.

    **Output:**

    - numpy array of shape (num_views, num_angles, 3, 3) of the output rotation matrix
    """
    num_views = batch_towards.shape[0]
    num_angles = len(batch_angles)
    batch_towards = np.repeat(batch_towards, num_angles, axis=0)
    batch_angles = np.tile(batch_angles, num_views)
    # print(batch_towards.shape)
    # print(batch_angles.shape)
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:, 1], axis_x[:, 0], zeros], axis=-1)
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    sin = np.sin(batch_angles)
    cos = np.cos(batch_angles)
    R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R1 = R1.reshape([-1, 3, 3])
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32).reshape(num_views, num_angles, 3, 3)


def generate_matrix(num_views=NUM_VIEWS, num_angles=NUM_ANGLES):
    views = generate_views(num_views)
    towards = -views
    angles = generate_angles(num_angles)
    return cross_viewpoint_params_to_matrix(towards, angles)


if __name__ == "__main__":
    import time

    views = generate_views(NUM_VIEWS)
    towards = -views
    angles = generate_angles(NUM_ANGLES)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(views)
    t1 = time.time()

    Rs = cross_viewpoint_params_to_matrix(towards, angles)

    t2 = time.time()

    for toward in towards:
        for angle in angles:
            viewpoint_params_to_matrix(toward, angle)

    t3 = time.time()
    print(
        "matrix time:{}, for time:{}, acceleration:{}".format(
            t2 - t1, t3 - t2, (t3 - t2) / (t2 - t1)
        )
    )

    Rs = Rs.reshape((-1, 3, 3))
    np.random.shuffle(Rs)
    # Rs = Rs[0:20]
    from vis import plot_gripper_pro_max

    geometry = []
    pcd = o3d.geometry.PointCloud()
    points = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=np.float32
    )
    pcd.points = o3d.utility.Vector3dVector(points)
    geometry.append(pcd)
    for R in Rs:
        geometry.append(
            plot_gripper_pro_max(
                np.array([0, 0, 0.1]), R, 0.02, 0.02, score=np.random.random()
            )
        )
    o3d.visualization.draw_geometries(geometry)
