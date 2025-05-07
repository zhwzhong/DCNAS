# -*- coding: utf-8 -*-

import scipy
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F


def pixel(tensor):
    return int(tensor.size(2) * tensor.size(3))


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = pixel(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = sum(self.log_s) * pixel(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

class FrequencyBranch(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyBranch, self).__init__()
        self.pre1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self.conv_amp = nn.Sequential(*[
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        ])

        self.conv_phase = nn.Sequential(*[
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        ])

        self.post = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, ms, pan):
        _, _, H, W = ms.shape
        amp_ms = torch.abs(torch.fft.fft2(self.pre1(ms) + 1e-8, norm='backward'))
        amp_pan = torch.abs(torch.fft.fft2(self.pre1(pan) + 1e-8, norm='backward'))

        phase_ms = torch.angle(torch.fft.fft2(amp_ms))
        phase_pan = torch.angle(torch.fft.fft2(amp_pan))

        amp_fuse = self.conv_amp(torch.cat([amp_ms, amp_pan], dim=1))
        pha_fuse = self.conv_phase(torch.cat([phase_ms, phase_pan], dim=1))

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        return self.post(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DualDomainInteraction(nn.Module):
    def __init__(self, in_channels):
        super(DualDomainInteraction, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, spatial_feat, freq_feat):
        diff = torch.abs(spatial_feat - freq_feat)
        spatial_att = self.spatial_attention(diff)
        enhanced_freq_feat = spatial_feat + spatial_att * freq_feat
        fused_feat = torch.cat([enhanced_freq_feat, spatial_feat], dim=1)
        channel_att = self.channel_attention(fused_feat)
        return self.conv1(channel_att * fused_feat)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class SFIB(nn.Module):
    def __init__(self, in_channels):
        super(SFIB, self).__init__()

        self.pan_process = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        ])
        self.spatial_process = nn.Sequential(*[
            InvBlock(DenseBlock, 2 * in_channels, in_channels),
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        ])
        self.fre_precess = FrequencyBranch(in_channels=in_channels)

        self.interaction = DualDomainInteraction(in_channels=in_channels)

    def forward(self, ms, pan):
        pan = self.pan_process(pan)
        spa_feature = self.spatial_process(torch.cat([ms, pan], 1))
        fre_feature = self.fre_precess(ms, pan)
        fuse_feature = self.interaction(fre_feature, spa_feature)
        return fuse_feature + ms


class SFINet(nn.Module):
    def __init__(self, in_channels, guidance_channels, num_features):
        super(SFINet, self).__init__()

        self.pan_head = nn.Conv2d(guidance_channels, num_features, kernel_size=1)
        self.ms_head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.pan_conv = nn.ModuleList([
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
        ])

        self.fuse_conv = nn.ModuleList([
            SFIB(in_channels=num_features),
            SFIB(in_channels=num_features),
            SFIB(in_channels=num_features),
            SFIB(in_channels=num_features),
            SFIB(in_channels=num_features),
        ])
        self.final_conv = nn.Conv2d(num_features * 5, in_channels, kernel_size=1)

    def forward(self, samples):
        l_ms, bms, pan = samples['img_lr'], samples['lr_up'], samples['img_rgb']
        pan_feature = self.pan_head(pan)
        ms_feature = self.ms_head(bms)

        out_features = []
        for i in range(5):
            pan_feature = self.pan_conv[i](pan_feature)
            ms_feature = self.fuse_conv[i](ms_feature, pan_feature)
            out_features.append(ms_feature)

        return {'img_out':self.final_conv(torch.cat(out_features, 1))}


def make_model(args):
    return SFINet(in_channels=args.in_channels, guidance_channels=args.guidance_channels, num_features=48)

# # Example usage
# if __name__ == "__main__":
#     from config import args
#     model = SFINet(in_channels=4, guidance_channels=1, num_features=32).cuda()
#     cd, cg, h, w = 4, 1, 480, 480
#
#     samples = {
#         'lr_up': torch.randn(1, cd, h, w).float().cuda(),
#         'img_rgb': torch.randn(1, cg, h, w).float().cuda(),
#         'lr_mask': torch.randn(1, cd, h // args.scale, w // args.scale).float().cuda(),
#         'img_lr': torch.randn(1, cd, h // args.scale, w // args.scale).float().cuda()
#     }
#     output = model(samples)
#     print(output.shape)

