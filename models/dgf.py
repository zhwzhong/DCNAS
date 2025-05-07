# -*- coding: utf-8 -*-

import torch
from torch import nn
from models.guided_filter import ConvGuidedFilter
from torchvision.transforms.functional import rgb_to_grayscale

class DGF(nn.Module):
    def __init__(self, args):
        super(DGF, self).__init__()
        self.args = args
        self.layers = ConvGuidedFilter(in_channels=1, num_features=64)

    def forward(self, sample):
        rgb, lr_up = sample['img_rgb'], sample['lr_up']
        rgb = rgb_to_grayscale(rgb, 1)
        out = self.layers(rgb, lr_up)
        return {'img_out': out}


def make_model(args): return DGF(args)