__author__ = "Minghao Gou"
__version__ = "1.0"

import torch
import torch.nn as nn


class Acc_v2(nn.Module):
    def __init__(self, thresh=0.5):
        super(Acc_v2, self).__init__()
        self.thresh = thresh

    def forward(self, batch_prob_map, batch_label, topK=20):
        batch_size = batch_prob_map.shape[0]
        acc_list = []
        for i in range(batch_size):
            prob_map = batch_prob_map[i]
            label = batch_label[i]
            true_mask = label == 1.0

            acc_true = torch.mean((prob_map[true_mask] > self.thresh).float())
            acc_false = torch.mean((prob_map[~true_mask] <= self.thresh).float())
            pred_true_mask = prob_map > self.thresh
            pred_true_num = torch.sum(pred_true_mask.float()).int()
            precision = torch.mean((label[pred_true_mask]).float())
            flatten_prob_map = torch.reshape(prob_map, (-1,))
            flatten_label = torch.reshape(label, (-1,))
            sorted_arg = torch.argsort(flatten_prob_map, descending=True)[
                : topK * batch_size
            ]
            label_at_arg = flatten_label[sorted_arg]
            topK_acc = torch.mean(label_at_arg)
            acc_list.append([acc_true, acc_false, precision, pred_true_num, topK_acc])
        acc_tensor = torch.tensor(acc_list).float()
        acc_true, acc_false, precision, pred_true_num, topK_acc = torch.mean(
            acc_tensor, dim=0
        )
        pred_true_num = pred_true_num.int()
        return acc_true, acc_false, precision, pred_true_num, topK_acc
