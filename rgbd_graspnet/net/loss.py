__author__ = "Minghao Gou"
__version__ = "1.0"
import numpy as np
import torch
import torch.nn as nn

from ..data.utils.generate_anchor_matrix import (
    NUM_VIEWS,
    NUM_ANGLES,
    generate_angles,
    generate_views,
)
from ..data.utils.transformation import NUM_GRID_Y, NUM_GRID_X


def generate_angle_vectors(N=NUM_ANGLES):
    """
    **Input:**

    - angles: numpy array of shape (n,)

    **Output:**

    - numpy array of the angles represented by vectors in xy plane
    """
    angles = generate_angles(N)
    return np.stack([np.cos(2 * angles), np.sin(2 * angles), np.zeros(N)]).T


class Class_Loss(nn.Module):
    @classmethod
    def View_Loss(cls, num_views=NUM_VIEWS):
        """
        **Input:**

        - num_views: int of number of views

        **Output:**

        - View_Loss instance
        """
        return cls(num_classes=num_views, generate_method=generate_views)

    @classmethod
    def Angle_Loss(cls, num_angles=NUM_ANGLES):
        """
        **Input:**

        - num_views: int of number of angles

        **Output:**

        - Angle_Loss instance
        """
        return cls(num_classes=num_angles, generate_method=generate_angle_vectors)

    def __init__(self, num_classes, generate_method, numpy=False):
        """
        **Input:**

        - num_classes: int of number of classes

        - generate_method: function of generating vectors function.

        - numpy: bool of whether the input and output format is numpy array
        """
        super(Class_Loss, self).__init__()
        self.num_classes = num_classes
        self.dim1_softmax = nn.Softmax(dim=1)
        self.generate_method = generate_method
        self.vectors = torch.tensor(self.generate_method(self.num_classes))
        self.loss_matrix = self.get_loss_matrix()  # np array
        self.loss_matrix_torch = torch.tensor(self.loss_matrix)  # torch.tensor
        self.numpy = numpy

    def forward(self, class_pred, class_label):
        """
        **Input:**

        - class_pred: torch.tensor of shape (batch_size, num_classes).

        - class_label: torch.tensor of shape (batch_size,).

        **Output:**

        - loss: mean loss of the class.
        """
        assert class_pred.device == class_label.device
        self.loss_matrix_torch = self.loss_matrix_torch.to(class_pred.device)
        loss_cols = torch.index_select(
            self.loss_matrix_torch, 0, class_label
        )  # (batch_size, num_class)
        loss_array = torch.sum(loss_cols * self.dim1_softmax(class_pred), dim=1)
        return torch.mean(loss_array)

    def get_mean_error(self, class_pred, class_label):
        """
        **Input:**

        - class_pred: torch.tensor of shape (batch_size, num_classes).

        - class_label: torch.tensor of shape (batch_size,).

        **Output:**

        - acc: torch.tensor of the mean angle difference.
        """
        assert class_pred.shape[0] == class_label.shape[0]
        assert class_pred.device == class_label.device
        self.vectors = self.vectors.to(class_pred.device)
        batch_size = class_pred.shape[0]
        # softmax is not needed here as the result of calculating argmax is the same.
        pred_indices = torch.argmax(class_pred, dim=1)
        angles_pred = torch.index_select(
            self.vectors, 0, pred_indices
        )  # (batch_size, 3)
        angles_label = torch.index_select(
            self.vectors, 0, class_label
        )  # (batch_size, 3)
        cos_diff = torch.sum(angles_pred * angles_label, dim=1)
        cos_diff[cos_diff >= 1.0] = 1.0
        cos_diff[cos_diff <= -1.0] = -1.0

        angles_diff = torch.acos(cos_diff)
        return torch.mean(angles_diff)

    def get_acc(self, class_pred, class_label, angle_type, thresh):
        """
        **Input:**

        - class_pred: torch.tensor of shape (batch_size, num_classes).

        - class_label: torch.tensor of shape (batch_size,).

        - angle_type: string of type of angle, 'angle' for rotation angle and 'view' for views.

        - thresh: the thresh of the angle which is consider correct.

        **Output:**

        - acc: torch.tensor of the correct rate.
        """
        assert class_pred.shape[0] == class_label.shape[0]
        assert class_pred.device == class_label.device
        self.vectors = self.vectors.to(class_pred.device)

        pred_indices = torch.argmax(class_pred, dim=1)

        angles_pred = torch.index_select(
            self.vectors, 0, pred_indices
        )  # (batch_size, 3)

        angles_label = torch.index_select(
            self.vectors, 0, class_label
        )  # (batch_size, 3)

        cos_diff = torch.sum(angles_pred * angles_label, dim=1)
        cos_diff[cos_diff >= 1.0] = 1.0
        cos_diff[cos_diff <= -1.0] = -1.0

        angles_diff = torch.acos(cos_diff)
        if angle_type == "angle":
            angles_diff = angles_diff / 2
        if angle_type == "view":
            pass
        angles_correct = angles_diff < thresh
        return torch.mean(angles_correct.float())

    def get_cos_matrix(self):
        """
        **Output:**

        - cos_matrix: np.array of shape (self.num_classes, self.num_classes) which is symmetric.

        - cos_matrix[i][j] is the cosine value between the i^{th} view and j^{th} view. Range: [-1, 1]
        """
        vectors = self.generate_method(self.num_classes)
        # views: (N, 3)
        cos_matrix = np.dot(vectors, np.transpose(vectors, [1, 0]))
        return cos_matrix

    def get_loss_matrix(self):
        """
        **Output:**

        - loss_matrix: np.array of shape (self.num_classes, self.num_classes) which is symmetric.

        - loss_matrix[i][j] is the cosine value between the i^{th} view and j^{th} view. Range: [0, 1]
        """
        cos_matrix = self.get_cos_matrix()
        loss_matrix = (-cos_matrix + 1.0) / 2
        return loss_matrix


View_Loss = Class_Loss.View_Loss
Angle_Loss = Class_Loss.Angle_Loss
