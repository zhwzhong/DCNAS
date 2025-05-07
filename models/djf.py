# -*- coding: utf-8 -*-

import torch
from torch import nn
from models.common import ConvBNReLU2D


class DJF(nn.Module):
    def __init__(self, args):
        super(DJF, self).__init__()
        self.depth_encoder = nn.Sequential(
            ConvBNReLU2D(in_channels=1, out_channels=96, kernel_size=9, stride=1, padding=2, act='ReLU'),
            ConvBNReLU2D(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=2, act='ReLU'),
            ConvBNReLU2D(in_channels=48, out_channels=3, kernel_size=5, stride=1, padding=2)
        )

        self.rgb_encoder = nn.Sequential(
            ConvBNReLU2D(in_channels=args.guidance_channels, out_channels=96, kernel_size=9, stride=1, padding=2, act='ReLU'),
            ConvBNReLU2D(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=2, act='ReLU'),
            ConvBNReLU2D(in_channels=48, out_channels=3, kernel_size=5, stride=1, padding=2)
        )

        self.decoder = nn.Sequential(
            ConvBNReLU2D(in_channels=6, out_channels=64, kernel_size=9, stride=1, padding=4, act='ReLU'),
            ConvBNReLU2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, act='ReLU'),
            ConvBNReLU2D(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, samples):
        lr, rgb, lr_up = samples['img_lr'], samples['img_rgb'], samples['lr_up']
        rgb_out = self.rgb_encoder(rgb)
        dep_out = self.depth_encoder(lr_up)
        out = self.decoder(torch.cat((rgb_out, dep_out), dim=1))
        return {'img_out': out}


def make_model(args): return DJF(args)