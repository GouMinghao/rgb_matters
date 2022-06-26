# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# Modified by Minghao Gou
# -----------------------------------------------------

import torch.nn as nn

# from .builder import SPPE
from .layers.DUC import DUC
from .layers.SE_Resnet import SEResnet


# @SPPE.register_module
class FastPose(nn.Module):
    conv_dim = 512

    def __init__(self, norm_layer=nn.BatchNorm2d, num_layers=50):
        super(FastPose, self).__init__()
        # self._preset_cfg = cfg['PRESET']

        self.preact = SEResnet(f"resnet{num_layers}")

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403

        assert num_layers in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{num_layers}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {
            k: v
            for k, v in x.state_dict().items()
            if k in self.preact.state_dict()
            and v.size() == self.preact.state_dict()[k].size()
        }
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 2048, upscale_factor=2, norm_layer=norm_layer)
        self.duc2 = DUC(512, 2048, upscale_factor=2, norm_layer=norm_layer)
        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(
            self.conv_dim, 360, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        # out = self.sigmoid(out)
        return out

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
