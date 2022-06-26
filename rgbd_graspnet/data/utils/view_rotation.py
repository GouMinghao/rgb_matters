__author__ = "Minghao Gou"
__version__ = "1.0"


import numpy as np
from .generate_anchor_matrix import (
    NUM_VIEWS,
    NUM_ANGLES,
)


def get_towards_and_angles(Rs, anchor_matrix):
    """
    **Input:**

    - Rs: numpy array of shape (n, 3, 3) of the rotation matrices.

    - anchor_matrix: numpy array of shape (num_views, num_angles, 3, 3) of the pre-defined rotation matrices.

    **Output:**

    - towards_index: numpy array of shape (n,) of the predefined towards index.

    - angles_index: numpy array of shape (n,) of the predifined angle index.
    """
    inv_anchor_matrix = np.linalg.inv(anchor_matrix)
    inv_anchor_matrix = inv_anchor_matrix[:, :, np.newaxis, :, :]
    inv_anchor_matrix = np.repeat(inv_anchor_matrix, Rs.shape[0], axis=2)
    matmul_result = np.matmul(inv_anchor_matrix, Rs)
    trace = np.trace(matmul_result, axis1=-2, axis2=-1)  # (120, 12, n)
    trace = trace.reshape(-1, trace.shape[-1])  # (1440, n)
    positions = np.argmax(trace, axis=0)

    towards_index = positions // NUM_ANGLES
    angles_index = positions % NUM_ANGLES
    return towards_index, angles_index
