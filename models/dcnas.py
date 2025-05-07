# -*- coding: utf-8 -*-

import random

import torch
import numpy as np
from torch import nn
from models.mobile import MBlock
from models.fusion import FusionBlock
from models.common import ConvBNReLU2D
from models.fix_dcnas import StaticModel


class FeatureInitialization(nn.Module):
    # 提取Depth和RGB的特征，变为64个通道
    def __init__(self, num_features, in_channels, guidance_channels, act, bias, norm):
        super(FeatureInitialization, self).__init__()

        self.depth_head = nn.Sequential(
            ConvBNReLU2D(in_channels, num_features, 3, padding=1, act=act, bias=bias, norm=norm),
            # ConvBNReLU2D(num_features, num_features, 3, padding=1, act=act, bias=bias, norm=norm)
        )
        self.guidance_head = nn.Sequential(
            ConvBNReLU2D(guidance_channels, num_features, 3, padding=1, act=act, bias=bias, norm=norm),
            # ConvBNReLU2D(num_features, num_features, 3, padding=1, act=act, bias=bias, norm=norm)
        )

    def forward(self, depth, guidance):
        return self.depth_head(depth), self.guidance_head(guidance)


class MBlocks(nn.Module):

    def __init__(self, num_blocks, in_channels, out_channels, stride, act, bias, norm, choice=None):    # choice: list
        super(MBlocks, self).__init__()

        self.mb_blocks = nn.ModuleList()
        self.fix_arch = choice is not None

        for i in range(num_blocks):
            self.mb_blocks.append(
                MBlock(in_channels, out_channels, stride, act, bias, norm, choice[i] if choice is not None else choice)
            )

    def forward(self, x, choice=None):
        res = x
        assert choice is not None or self.fix_arch, 'Please Select One Op...'
        for i, select_op in enumerate(self.mb_blocks):
            x = select_op(x, choice[i] if choice is not None else choice)
        return x + res

class UpSample(nn.Module):
    def __init__(self, scale, num_features, bias, keep_dim=True):
        super(UpSample, self).__init__()

        self.layers = nn.Sequential(
            ConvBNReLU2D(num_features, (num_features * 4) if keep_dim else (num_features * 2), 1, bias=bias),
            nn.PixelShuffle(upscale_factor=scale)
        )
    def forward(self, x):
        return self.layers(x)


class DownSample(nn.Module):
    def __init__(self, scale, num_features, bias, keep_dim=True):
        super(DownSample, self).__init__()

        self.dep_layers = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=scale),
            ConvBNReLU2D(num_features * 4, num_features if keep_dim else (num_features * 2), 1, bias=bias),
        )
        self.rgb_layers = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=scale),
            ConvBNReLU2D(num_features * 4, num_features if keep_dim else (num_features * 2), 1, bias=bias),
        )

    def forward(self, dep, rgb):
        return self.dep_layers(dep), self.rgb_layers(rgb)


class Tail(torch.nn.Module):
    def __init__(self, num_features, out_channels, act, bias):
        super(Tail, self).__init__()
        self.layers = nn.Sequential(
            # ConvBNReLU2D(num_features, num_features, 3, padding=1, act=act, bias=bias),
            ConvBNReLU2D(num_features, out_channels, 3, padding=1, act=act, bias=bias),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class DCNAS(nn.Module):
    def __init__(self, args, arch=None):
        super(DCNAS, self).__init__()
        self.args = args

        self.dep_encoder = nn.ModuleList()
        self.rgb_encoder = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        self.fusion_layer = nn.ModuleList()
        self.reconstruction = nn.ModuleList()
        self.fix_arch = (arch is not None)
        # print(self.fix_arch, arch)
        self.FeatureInitialization = FeatureInitialization(
            num_features=args.num_features, in_channels=args.in_channels,
            guidance_channels=args.guidance_channels, act=args.act, bias=args.bias, norm=args.norm
        )
        # args.num_features, args.num_features * 2, args.num_features * 4, args.num_features * 8
        num_features = [args.num_features * pow(2, i) for i in range(args.num_stages)]

        for stage in range(args.num_stages):
            i_stage = args.num_stages - stage - 1
            self.dep_encoder.append(
                MBlocks(args.num_blocks, num_features[stage], num_features[stage], 1, args.act, bias=args.bias,
                        norm=args.norm, choice=arch['dep_encoder'][stage] if self.fix_arch else None)
            )
            self.rgb_encoder.append(
                MBlocks(args.num_blocks, num_features[stage], num_features[stage], 1, args.act, bias=args.bias,
                        norm=args.norm, choice=arch['rgb_encoder'][stage] if self.fix_arch else None)
            )
            self.reconstruction.append(
                MBlocks(args.num_blocks, num_features[i_stage], num_features[i_stage], 1, args.act,  bias=args.bias,
                        norm=args.norm, choice=arch['rec_decoder'][stage] if self.fix_arch else None)
            )
            self.fusion_layer.append(
                FusionBlock(num_features[i_stage], choice=arch['fuse_op'][stage] if self.fix_arch else None)
            )
            if stage != args.num_stages - 1:
                self.down_sample.append(DownSample(scale=2, num_features=num_features[stage], bias=args.bias, keep_dim=False))
                self.up_sample.append(UpSample(scale=2, num_features=num_features[i_stage], bias=args.bias, keep_dim=False))

        self.middle_layer = ConvBNReLU2D(num_features[-1], num_features[-1], 1, act=args.act, bias=args.bias)
        self.tail = Tail(num_features[0], args.in_channels, args.act, bias=args.bias)

    def forward(self, samples=None, arch=None, dep=None, rgb=None):

        if samples is not None:
            dep, rgb = samples['lr_up'], samples['img_rgb']
        lr_up = dep
        dep, rgb = self.FeatureInitialization(dep, rgb)

        dep_out = []
        rgb_out = []

        for index in range(self.args.num_stages):

            rgb = self.rgb_encoder[index](rgb, None if self.fix_arch else arch['rgb_encoder'][index])
            rgb_out.append(rgb)

            dep = self.dep_encoder[index](dep, None if self.fix_arch else arch['dep_encoder'][index])
            dep_out.append(dep)

            if index != self.args.num_stages - 1:
                dep, rgb = self.down_sample[index](dep, rgb)

        dep_out, rgb_out = list(reversed(dep_out)), list(reversed(rgb_out))

        rec_out = self.middle_layer(dep_out[0])
        for index in range(self.args.num_stages):
            fuse_out = self.fusion_layer[index](dep_out[index], rgb_out[index], None if self.fix_arch else arch['fuse_op'][index])
            rec_out = self.reconstruction[index](rec_out + fuse_out, None if self.fix_arch else arch['rec_decoder'][index])
            if index != self.args.num_stages - 1:
                rec_out = self.up_sample[index](rec_out)

        # return self.tail(rec_out) + lr_up
        return {'img_out': self.tail(rec_out) + lr_up}


def make_model(args, arch):
    if arch is not None:
        return StaticModel(args, arch)
    return DCNAS(args, arch)

