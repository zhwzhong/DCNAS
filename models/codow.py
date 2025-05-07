# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.common import ConvBNReLU2D


def make_model(args): return Codon(args)


class Codon(nn.Module):
    def __init__(self, args):
        super(Codon, self).__init__()

        self.args = args

        base_filter = 64

        self.conv1_d = ConvBNReLU2D(args.in_channels, base_filter, 3, 1, 1, act='PReLU')
        self.conv1_c = ConvBNReLU2D(args.guidance_channels, base_filter, 3, 1, 1, act='PReLU')

        self.rmc1 = RMC1(64)

        self.conv2 = ConvBNReLU2D(base_filter * 2, base_filter, 3, 1, 1, act='PReLU')
        self.rmc2 = RMC2(64)
        self.conv3 = ConvBNReLU2D(base_filter, base_filter, 3, 1, 1, act='PReLU')
        self.conv4 = ConvBNReLU2D(base_filter, args.in_channels, 3, 1, 1)

    def forward(self, samples):
        lr, rgb, lr_up = samples['img_lr'], samples['img_rgb'], samples['lr_up']
        out = self.net(lr, rgb, lr_up)
        return out

    # @torchsnooper.snoop()
    def forward_chop(self, depth, rgb, up, scale, shave=10, min_size=50000):
        n_GPUs = 1
        b, c, h, w = depth.size()
        b2, c2, h2, w2 = rgb.size()
        #############################################
        # adaptive shave
        shave_scale = 4
        # max shave size
        shave_size_max = 4
        # get half size of the hight and width
        h_half, w_half = h // 2, w // 2
        h2_half, w2_half = h2 // 2, w2 // 2
        # mod
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        mod_h2, mod_w2 = h2_half // shave_scale, w2_half // shave_scale
        # ditermine midsize along height and width directions
        h_size = mod_h * shave_scale + shave_size_max
        w_size = mod_w * shave_scale + shave_size_max
        h2_size = scale * h_size
        w2_size = scale * w_size
        depth_lr_list = [
            depth[:, :, 0:h_size, 0:w_size],
            depth[:, :, 0:h_size, (w - w_size):w],
            depth[:, :, (h - h_size):h, 0:w_size],
            depth[:, :, (h - h_size):h, (w - w_size):w]]
        rgb_lr_list = [
            rgb[:, :, 0:h2_size, 0:w2_size],
            rgb[:, :, 0:h2_size, (w2 - w2_size):w2],
            rgb[:, :, (h2 - h2_size):h2, 0:w2_size],
            rgb[:, :, (h2 - h2_size):h2, (w2 - w2_size):w2]]
        up_list = [
            up[:, :, 0:h2_size, 0:w2_size],
            up[:, :, 0:h2_size, (w2 - w2_size):w2],
            up[:, :, (h2 - h2_size):h2, 0:w2_size],
            up[:, :, (h2 - h2_size):h2, (w2 - w2_size):w2]]
        if w2_size * h2_size < min_size:
            out_sr_list = []
            for i in range(0, 4, n_GPUs):
                depth_lr_batch = torch.cat(depth_lr_list[i:(i + n_GPUs)], dim=0)
                rgb_lr_batch = torch.cat(rgb_lr_list[i:(i + n_GPUs)], dim=0)
                up_batch = torch.cat(up_list[i:(i + n_GPUs)], dim=0)

                out_sr_batch = self.net(depth_lr_batch, rgb_lr_batch, up_batch)[-1]

                out_sr_list.append(out_sr_batch)
                # del out_sr_batch

        else:
            out_sr_list = [
                self.forward_chop(patch, patch2, patch3, scale=scale, shave=shave, min_size=min_size)[-1] \
                for patch, patch2, patch3 in zip(depth_lr_list, rgb_lr_list, up_list)
            ]

        # h2, w2 = scale * h2_size, scale * w2_size
        # h2_half, w2_half = scale * h2_half, scale * w2_half
        # h2_size, w2_size = scale * h2_size, scale * w2_size
        # shave *= scale

        output = torch.tensor(rgb.data.new(b, c, h2, w2))
        outmask = torch.zeros_like(output)
        output[:, :, 0:h2_half, 0:w2_half] \
            = out_sr_list[0][:, :, 0:h2_half, 0:w2_half]
        output[:, :, 0:h2_half, (w2 - w2_half):w2] \
            = out_sr_list[1][:, :, 0:h2_half, (w2_size - w2_half):w2_size]

        output[:, :, (h2 - h2_half):h2, 0:w2_half] \
            = out_sr_list[2][:, :, (h2_size - h2_half):h2_size, 0:w2_half]
        output[:, :, (h2 - h2_half):h2, (w2 - w2_half):w2] \
            = out_sr_list[3][:, :, (h2_size - h2_half):h2_size, (w2_size - w2_half):w2_size]

        out_mask0 = torch.ones_like(out_sr_list[0])
        out_mask1 = torch.ones_like(out_sr_list[1])
        out_mask2 = torch.ones_like(out_sr_list[2])
        out_mask3 = torch.ones_like(out_sr_list[3])

        outmask[:, :, 0:h2_half, 0:w2_half] \
            += out_mask0[:, :, 0:h2_half, 0:w2_half]
        outmask[:, :, 0:h2_half, (w2 - w2_half):w2] \
            += out_mask1[:, :, 0:h2_half, (w2_size - w2_half):w2_size]
        outmask[:, :, (h2 - h2_half):h2, 0:w2_half] \
            += out_mask2[:, :, (h2_size - h2_half):h2_size, 0:w2_half]
        outmask[:, :, (h2 - h2_half):h2, (w2 - w2_half):w2] \
            += out_mask3[:, :, (h2_size - h2_half):h2_size, (w2_size - w2_half):w2_size]
        outmask += (outmask == 0.).float()
        # output = output / outmask
        return [output]

    # @torchsnooper.snoop()
    def net(self, depth, rgb, lr_up):
        d = self.conv1_d(lr_up)
        r = self.conv1_c(rgb)

        d, r = self.rmc1(d, r)

        y = torch.concat((d, r), 1)
        y = self.conv2(y)
        y = self.rmc2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        out = y + lr_up

        return {'img_out': out}


class CAC(torch.nn.Module):
    def __init__(self, num_filter):
        super(CAC, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)
        self.g_max_pooling1 = torch.nn.AdaptiveMaxPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))
        self.act_1 = torch.nn.ReLU(True)
        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=round(num_filter / 2))
        self.act_2 = torch.nn.Sigmoid()

        self.cov_5 = nn.Conv2d(2, 1, 5, 1, 2, bias=True)

    # @torchsnooper.snoop()
    def forward(self, x):
        x_avg = self.g_aver_pooling1(x)
        x_avg = x_avg.view(x_avg.size(0), -1)
        x_avg2 = self.fc1(x_avg)
        act1 = self.act_1(x_avg2)
        x_avg2 = self.fc2(act1)

        x_max = self.g_max_pooling1(x)
        x_max = x_max.view(x_max.size(0), -1)
        x_max2 = self.fc1(x_max)
        act2 = self.act_1(x_max2)
        x_max2 = self.fc2(act2)

        channel_a = self.act_2(x_avg2 + x_max2)
        channel_a = channel_a.view(act2.size(0), channel_a.size(1), 1, 1)

        y_avg = torch.unsqueeze(torch.mean(x, dim=1), 1)
        y_max, _ = torch.max(x, dim=1)
        y_max = torch.unsqueeze(y_max, 1)
        y = torch.cat((y_avg, y_max), 1)
        y = self.cov_5(y)
        space_a = self.act_2(y)

        out = channel_a * space_a
        # out = x + x*out
        return out


class MC(torch.nn.Module):
    def __init__(self, num_filter, bias=True, activation='prelu'):
        super(MC, self).__init__()

        self.cov1_3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.cov1_5 = nn.Conv2d(64, 64, 5, 1, 2, bias=True)
        self.act_1 = torch.nn.ReLU(True)
        self.cov2 = nn.Conv2d(2 * num_filter, 2 * num_filter, 5, 1, 2, bias=True)
        self.out_cov = nn.Conv2d(2 * num_filter, num_filter, 1, 1, 0, bias=True)

    # @torchsnooper.snoop()
    def forward(self, x):
        x_3 = self.cov1_3(x)
        x_3 = self.act_1(x_3)
        x_5 = self.cov1_5(x)
        x_5 = self.act_1(x_5)

        y = torch.cat((x_3, x_5), 1)

        y = self.cov2(y)
        out = self.out_cov(y)
        return out


class RMC1(torch.nn.Module):
    def __init__(self, num_filter):
        super(RMC1, self).__init__()

        self.mc_d = MC(num_filter)
        self.mc_c = MC(num_filter)
        self.cac = CAC(2 * num_filter)

    # @torchsnooper.snoop()
    def forward(self, depth, rgb):
        d, r = self.recur(depth, rgb, depth, rgb, 4)
        return d, r

    # @torchsnooper.snoop()
    def recur(self, depth, rgb, depth2, rgb2, index):
        d = self.mc_d(depth)
        r = self.mc_c(rgb)
        att = torch.cat((r, d), 1)
        att = self.cac(att)
        d = d * att + depth2
        r = r * att + rgb2
        if index == 0:
            return d, r
        else:
            d2, r2 = self.recur(d, r, depth2, rgb2, index - 1)
            d = depth2 + d2
            r = rgb2 + r2
            return d, r


class RMC2(torch.nn.Module):
    def __init__(self, num_filter):
        super(RMC2, self).__init__()

        self.mc = MC(num_filter)

    def forward(self, x):
        y = self.recur(x, x, 2)
        return y

    def recur(self, x, x2, index):
        y = self.mc(x)
        y = y + x2
        if index == 0:
            return y
        else:
            y2 = self.recur(y, x2, index - 1)
            y = y2 + x2
            return y
