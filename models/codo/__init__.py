# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   __init__.py.py
@Time    :   2023/4/21 09:31
@Desc    :
"""

from torch import nn
from . import CODON_x4, CODON_x8, CODON_x16
from torchvision.transforms.functional import rgb_to_grayscale

class Model(nn.Module):
    def __init__(self, scale):
        super(Model, self).__init__()

        if scale == 4:
            self.net = CODON_x4.CODONNet()
        if scale == 8:
            self.net = CODON_x8.CODONNet()
        if scale == 16:
            self.net = CODON_x16.CODONNet()

    def forward(self, samples):
        x, y = samples['lr_up'], rgb_to_grayscale(samples['img_rgb'], 1)

        return {'img_out': self.net(x, y)}


def make_model(args):
    return Model(args.scale)