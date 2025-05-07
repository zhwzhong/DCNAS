# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   pmban.py
@Time    :   2020/7/31 12:03
@Desc    :
"""
from torch import nn
from torch.nn.functional import pad
from models import pmpanet_x2, pmpanet_x4, pmpanet_x8, pmpanet_x16

def make_model(args): return PMBAN(args)


def tensor_pad(img, scale=32):
    h, w = img.size()[2:]

    pad_h = scale - h % scale
    pad_w = scale - w % scale

    padding = [pad_w, 0, pad_h, 0]
    img = pad(img, padding, "constant", 0)

    return img, pad_h, pad_w

class PMBAN(nn.Module):
    def __init__(self, args):
        super(PMBAN, self).__init__()
        self.args = args

        if self.args.scale == 2:
            self.model = pmpanet_x2.Net(num_channels=1, base_filter=64, feat=256, num_stages=3, scale_factor=2)
        elif self.args.scale == 4:
            self.model = pmpanet_x4.Net(num_channels=1, base_filter=64, feat=256, num_stages=3, scale_factor=4)
        elif self.args.scale == 8:
            self.model = pmpanet_x8.Net(num_channels=1, base_filter=64, feat=256, num_stages=3, scale_factor=8)
        else:
            self.model = pmpanet_x16.Net(num_channels=1, base_filter=64, feat=256, num_stages=3, scale_factor=16)
    def forward(self, samples):

        lr, rgb = samples['img_lr'], samples['img_rgb']
        lr, _, _ = tensor_pad(lr, 4)
        if rgb.size()[1] == 1:
            rgb = rgb.repeat(1, 3, 1, 1)
        rgb, h, w = tensor_pad(rgb, 4 * self.args.scale)
        return {'img_out': self.model(rgb, lr)[:, :, h:, w: ]}
