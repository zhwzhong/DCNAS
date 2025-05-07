from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from math import sqrt

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CHANNEL(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CHANNEL, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels // 2)
            )
        self.pool_types = pool_types

    def forward(self, x):
        shape_a = torch.zeros(x.shape[0], x.shape[1]//2, x.shape[2], x.shape[3])
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
                # print(channel_att_raw.shape)
            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
                # print(channel_att_raw.shape)
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(shape_a)
        return scale

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):

    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SPATIAL(nn.Module):
    def __init__(self):
        super(SPATIAL, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return scale


class CODONNet(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(CODONNet, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()

    def forward(self, x, y):      #x深度图 y彩色图
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):  #网络一共五层MC
            out_MC_R1 = self.relu(self.conv1(out))
            out_MC_P1_c = self.relu(self.conv5(out_c))

            out_MC_P1 = self.relu(self.conv2(out))
            out_MC_R1_c = self.relu(self.conv4(out_c))

            out_MC_stage = torch.cat((out_MC_R1, out_MC_P1), 1)
            out_MC_stage_c = torch.cat((out_MC_R1_c, out_MC_P1_c), 1)
            out_MC_R2 = self.relu(self.conv3(out_MC_stage))
            out_MC_R2_c = self.relu(self.conv6(out_MC_stage_c))
            out_c = self.confuse_c(out_MC_R2_c)
            out = self.confuse(out_MC_R2)
            CAC_cat = torch.cat((out_c, out), 1)    #Fcat
            if _ == 0:
                CAC_channel = self.attention_c0(CAC_cat)
                CAC_spatial = self.attention_s0(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 1:
                CAC_channel = self.attention_c1(CAC_cat)
                CAC_spatial = self.attention_s1(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 2:
                CAC_channel = self.attention_c2(CAC_cat)
                CAC_spatial = self.attention_s2(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 3:
                CAC_channel = self.attention_c3(CAC_cat)
                CAC_spatial = self.attention_s3(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 4:
                CAC_channel = self.attention_c4(CAC_cat)
                CAC_spatial = self.attention_s4(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):      #MC是一次循环
            out_fuse_MC_R1 = self.relu(self.conv8(out_fuse))
            out_fuse_MC_P1 = self.relu(self.conv9(out_fuse))
            out_fuse_MC_stage = torch.cat((out_fuse_MC_R1, out_fuse_MC_P1), 1)
            out_fuse_MC_R2 = self.relu(self.conv10(out_fuse_MC_stage))
            out_fuse = self.confuse_fuse(out_fuse_MC_R2)
            out_fuse = torch.add(out_fuse, fuse)   #每次MC后add residual
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)
        return out_fuse_final
