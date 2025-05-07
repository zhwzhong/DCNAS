# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from models.common import ConvBNReLU2D, GateConv2D
from models.guided_filter import ConvGuidedFilter

# Skip RGB 会导致参数量不一致的问题

block_dict = {
    'skip':
        lambda num_features: Skip(rgb=True, dep=True),
	'cat':
        lambda num_features: Cat(num_features),
    'sum':
        lambda num_features: Sum(),
    'energy':
        lambda num_features: Energy(),
    'mmaf':
        lambda num_features: MMAF(num_features),
	'skip_dep':
        lambda num_features: Skip(rgb=False, dep=True),
    'skip_rgb':
        lambda num_features: Skip(rgb=True, dep=False),
    # 'GF':
    #     lambda num_features: GF(num_features),
}

block_keys = list(block_dict.keys())

class Cat(nn.Module):
    def __init__(self, num_features):
        super(Cat, self).__init__()
        self.compress_layer = ConvBNReLU2D(in_channels=num_features * 2, out_channels=num_features, kernel_size=1, bias=False)

    def forward(self, dep, rgb):
        return self.compress_layer(torch.cat((dep, rgb), dim=1))


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()
    def forward(self, dep, rgb):
        return (dep + rgb).div(2)


class GF(nn.Module):
    def __init__(self, num_features):
        super(GF, self).__init__()
        self.layers = ConvGuidedFilter(in_channels=num_features, num_features=64, radius=5)

    def forward(self, dep, rgb):

        out = self.layers(rgb, dep)
        return out

# FCFR-Net: Feature Fusion based Coarse-to-Fine Residual Learning for Depth Completion
class Energy(nn.Module):
    def __init__(self):
        super(Energy, self).__init__()
        self.unfold = nn.Unfold(kernel_size=5, padding=2, stride=1)

    def forward(self, dep, rgb):
        b, c, h, w = dep.size()
        dep_weight = self.unfold(dep).view(b, c, -1, h, w)
        dep_energy = torch.einsum('bckhw, bckhw->bchw', [dep_weight, dep_weight])

        rgb_weight = self.unfold(rgb).view(b, c, -1, h, w)
        rgb_energy = torch.einsum('bckhw, bckhw->bchw', [rgb_weight, rgb_weight])

        return torch.where(dep_energy > rgb_energy, dep, rgb)

# Attention Based Fusion

# class Skip(nn.Module):
#     def __init__(self, rgb=True, dep=True):
#         super(Skip, self).__init__()
#         self.rgb = rgb
#         self.dep = dep
#     def forward(self, rgb, dep):
#         if self.rgb and self.dep:
#             return rgb * 0 + dep * 0
#         elif self.rgb:
#             return dep + rgb * 0
#         else:
#             return rgb + dep * 0

class Skip(nn.Module):
    def __init__(self, rgb=True, dep=True):
        super(Skip, self).__init__()
        self.rgb = rgb
        self.dep = dep
    def forward(self, dep, rgb):
        if self.rgb and self.dep:
            return 0
        elif self.rgb:
            return dep
        else:
            return rgb


def variance_pool(x):
    my_mean = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    return (x - my_mean).pow(2).mean(dim=3, keepdim=False).mean(dim=2, keepdim=False).view(x.size()[0], x.size()[1], 1, 1)


def pool_func(x, pool_type=None):
    b, c = x.size()[:2]
    if pool_type == 'avg':
        ret = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    elif pool_type == 'max':
        ret = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    elif pool_type == 'lp':
        ret = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    else:
        ret = variance_pool(x)
    return ret.view(b, c)


class MMAB(nn.Module):
    def __init__(self, num_features, reduction_ratio=4):
        super(MMAB, self).__init__()

        self.squeeze = ConvBNReLU2D(in_channels=num_features * 2, out_channels=num_features * 2 // reduction_ratio,
                                    kernel_size=3, act='PReLU', padding=1)

        self.excitation1 = ConvBNReLU2D(in_channels=num_features * 2 // reduction_ratio, out_channels=num_features,
                                        kernel_size=1, act='Sigmoid')
        self.excitation2 = ConvBNReLU2D(in_channels=num_features * 2 // reduction_ratio, out_channels=num_features,
                                        kernel_size=1, act='Sigmoid')

    def forward(self, depth, guidance):
        fuse_feature = self.squeeze(torch.cat((depth, guidance), 1))
        fuse_statistic = pool_func(fuse_feature, 'avg') + pool_func(fuse_feature)
        squeeze_feature = fuse_statistic.unsqueeze(2).unsqueeze(3)
        depth_out = self.excitation1(squeeze_feature)
        guidance_out = self.excitation2(squeeze_feature)
        return (depth_out * depth).div(2), (guidance_out * guidance).div(2)


class MMAF(nn.Module):
    def __init__(self, num_features, reduction_ratio=4):
        super(MMAF, self).__init__()

        self.filter_conv = GateConv2D(num_features=num_features)
        self.filter_conv1 = GateConv2D(num_features=num_features)
        self.attention_layer = MMAB(num_features=num_features, reduction_ratio=reduction_ratio)

    def forward(self, depth, guide):
        guide = self.filter_conv(guide)
        depth = self.filter_conv1(depth)
        depth, guide = self.attention_layer(depth=depth, guidance=guide)
        return depth + guide

# net = Cat(64)
# print(sum(p.numel() for p in net.parameters()))


class FusionBlock(nn.Module):
     def __init__(self, num_features, choice=None):
         super(FusionBlock, self).__init__()
         self.fix_arch = False
         self.fuse_layers = nn.ModuleList()
         self.num_operators = len(block_dict)
         if choice is not None:
             self.fix_arch = True
             self.fuse_layers = block_dict[block_keys[choice]](num_features)
         else:
             for idx, key in enumerate(block_keys):
                 self.fuse_layers.append(block_dict[key](num_features))

     def forward(self, dep, rgb, choice=None):
         assert choice is not None or self.fix_arch, 'Please Select One Op...'
         return self.fuse_layers(dep, rgb) if self.fix_arch else self.fuse_layers[choice](dep, rgb)
