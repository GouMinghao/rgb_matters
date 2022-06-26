__author__ = "Minghao Gou"
__version__ = "1.0"

import torch.nn as nn
import torch

from .fastpose import FastPose


class RGBNormalNet(nn.Module):
    def __init__(
        self,
        norm_layer=nn.BatchNorm2d,
        num_layers=50,
        use_normal=True,
        normal_only=False,
    ):
        super(RGBNormalNet, self).__init__()
        self.use_normal = use_normal
        self.normal_only = normal_only
        assert self.normal_only is False or self.use_normal is True
        if not self.normal_only:
            self.rgb_net = FastPose(norm_layer=norm_layer, num_layers=num_layers)
        if self.use_normal:
            self.normal_net = FastPose(norm_layer=norm_layer, num_layers=num_layers)
        if self.use_normal and not self.normal_only:
            self.post_conv = nn.Conv2d(720, 360, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, normal):
        if not self.normal_only:
            rgb_feature = self.rgb_net(rgb)
        if self.use_normal:
            normal_feature = self.normal_net(normal)
        if self.normal_only:
            return normal_feature
        elif self.use_normal:
            cat_feature = torch.cat((rgb_feature, normal_feature), dim=1)
            out = self.post_conv(cat_feature)
            return out
        else:
            return rgb_feature
