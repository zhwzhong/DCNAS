# -*- coding: utf-8 -*-

import torch
from torch import nn
from models import common
from models.common import ConvBNReLU2D, get_act
from torch.nn.functional import interpolate, softmax

# feature = []


# 测试不同的金字塔网络

def make_model(args): return DAGF(args)


class Head(nn.Module):
    def __init__(self, num_features, act, expand_ratio=1, guide_channels=3, in_channels=1):
        super(Head, self).__init__()

        self.depth_head = nn.Sequential(
            ConvBNReLU2D(in_channels=in_channels, out_channels=num_features * expand_ratio, kernel_size=3, padding=1),
            ConvBNReLU2D(in_channels=num_features * expand_ratio, out_channels=num_features, kernel_size=1, act=act)
        )

        self.guide_head = nn.Sequential(
            ConvBNReLU2D(in_channels=guide_channels, out_channels=num_features * expand_ratio, kernel_size=3, padding=1),
            ConvBNReLU2D(in_channels=num_features * expand_ratio, out_channels=num_features, kernel_size=1, act=act)
        )

    def forward(self, depth_img, guide_img):
        return self.depth_head(depth_img), self.guide_head(guide_img)

class PyModel(nn.Module):
    def __init__(self, num_features, out_channels=1):
        super(PyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.res_scale = common.Scale(0)

    def forward(self, inputs, add_feature):
        out = self.layer1(inputs)
        if add_feature is not None:
            out = self.layer3(self.res_scale(interpolate(add_feature, scale_factor=2, mode='nearest')) + out)
        return out, self.layer2(out)

class ResNet(nn.Module):
    def __init__(self, num_features, act, norm):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(*[
            ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, act=act, norm=norm),
            ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, norm=norm)
        ])
        self.act = get_act(act=act)

    def forward(self, input_feature):
        return self.act(self.layers(input_feature) + input_feature)

class FuseBlock(nn.Module):
    def __init__(self, num_feature, act, norm, kernel_size, num_res, scale=2):
        super(FuseBlock, self).__init__()

        self.scale = scale
        self.num = kernel_size * kernel_size

        self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))

        self.depth_kernel = nn.Sequential(
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=1, act=act, norm=norm),
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size ** 2, kernel_size=1)
        )

        self.guide_kernel = nn.Sequential(
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=1, act=act, norm=norm),
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size ** 2, kernel_size=1)
        )

        self.pix_shf = nn.PixelShuffle(upscale_factor=scale)

        self.weight_net = nn.Sequential(
            ConvBNReLU2D(in_channels=num_feature * 2, out_channels=num_feature, kernel_size=3, padding=1,
                                act=act, norm='Adaptive'),
            TUnet(num_features=num_feature, act=act, norm='Adaptive'),
            ConvBNReLU2D(in_channels=num_feature, out_channels=1, kernel_size=3, act=act,
                                padding=1, norm='Adaptive'),
        )
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)
        self.inputs_conv = nn.Sequential(*[
            ResNet(num_features=num_feature, act=act, norm=norm) for _ in range(num_res)])

    def forward(self, depth, guide, inputs, ret_kernel=False):
        b, c, h, w = inputs.size()
        h_, w_ = h * self.scale, w * self.scale
        weight_map = self.weight_net(torch.cat((depth, guide), 1))  # wu Softmax

        depth_kernel = self.depth_kernel(depth)
        guide_kernel = self.guide_kernel(guide)


        depth_kernel = softmax(depth_kernel, dim=1)
        guide_kernel = softmax(guide_kernel, dim=1)

        fuse_kernel = weight_map * depth_kernel + (1 - weight_map) * guide_kernel


        fuse_kernel = torch.tanh(fuse_kernel) / (self.aff_scale_const + 1e-8)

        abs_kernel = torch.abs(fuse_kernel)
        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4

        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0

        fuse_kernel = fuse_kernel / abs_kernel_sum

        inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')

        unfold_inputs = self.unfold(inputs_up).view(b, c, -1, h_, w_)
        out = torch.einsum('bkhw, bckhw->bchw', [fuse_kernel, unfold_inputs])
        if ret_kernel:
            return out, fuse_kernel, weight_map
        return out


class InitLayer(nn.Module):
    def __init__(self, in_channels, num_features, flag=0):
        super(InitLayer, self).__init__()

        self.flag = flag

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_features, padding=1, kernel_size=3),
            nn.PReLU()
        )

        if flag == 0:
            self.layer2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, padding=1, kernel_size=3)

        else:
            self.layer2 = nn.Conv2d(in_channels=2 * num_features, out_channels=num_features, padding=1, kernel_size=3)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_features, padding=1, kernel_size=3),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, padding=1, kernel_size=3)
        )

    def forward(self, inputs, lr):
        out = self.layer1(inputs)
        if self.flag == 0:
            out = self.layer2(out)
        else:
            out = self.layer2(torch.cat((out, lr), dim=1))
        return out

class TUnet(nn.Module):
    def __init__(self, num_features, act, norm):
        super(TUnet, self).__init__()
        self.up_sample = nn.PixelShuffle(2)
        self.down_sample = common.invPixelShuffle(2)
        self.encoder_1 = ConvBNReLU2D(in_channels=num_features, out_channels=num_features // 2, kernel_size=3, padding=1, act=act, norm=norm)
        self.encoder_2 = ConvBNReLU2D(in_channels=num_features * 2, out_channels=num_features, kernel_size=3, padding=1, act=act, norm=norm)
        self.feature_transform = nn.Sequential(
            ConvBNReLU2D(in_channels=num_features * 4, out_channels=num_features * 4, kernel_size=3, padding=1, act=act, norm=norm),
            ConvBNReLU2D(in_channels=num_features * 4, out_channels=num_features * 4, kernel_size=3, padding=1, act=act, norm=norm)
        )

        self.decoder_1 = ConvBNReLU2D(in_channels=num_features * 4, out_channels=num_features * 4 * 2, kernel_size=3, padding=1, act=act, norm=norm)
        self.decoder_2 = ConvBNReLU2D(in_channels=num_features * 2, out_channels=num_features * 2 * 2, kernel_size=3, padding=1, act=act, norm=norm)

    def forward(self, inputs):
        enc_out_1 = self.down_sample(self.encoder_1(inputs))
        enc_out_2 = self.down_sample(self.encoder_2(enc_out_1))
        feature_out = self.feature_transform(enc_out_2) + enc_out_2
        dec_out_1 = self.up_sample(self.decoder_1(feature_out))
        dec_out_2 = self.up_sample(self.decoder_2(dec_out_1 + enc_out_1))
        return dec_out_2 + inputs

class DAGF(nn.Module):
    def __init__(self, args):
        super(DAGF, self).__init__()
        self.args = args
        self.num_pyramid = args.num_pyramid
        self.head = Head(
            num_features=args.num_features, expand_ratio=1, act=args.act, guide_channels=args.guidance_channels,
            in_channels=args.in_channels
        )
        self.depth_pyramid = nn.ModuleList([
            common.DownSample(num_features=args.num_features, act=args.act, norm=args.norm) for _ in
            range(self.num_pyramid - 1)])

        self.guide_pyramid = nn.ModuleList([
            common.DownSample(num_features=args.num_features, act=args.act, norm=args.norm) for _ in
            range(self.num_pyramid - 1)])

        self.up_sample = nn.ModuleList([
            FuseBlock(num_feature=args.num_features, act=args.act, norm=args.norm, kernel_size=args.filter_size,
                      num_res=args.num_res, scale=2) for _ in range(self.num_pyramid)
        ])

        self.init_conv = nn.ModuleList(
            [InitLayer(args.in_channels, num_features=args.num_features, flag=i) for i in range(args.num_pyramid)])

        self.p_layers = nn.ModuleList([
            PyModel(args.num_features, out_channels=1) for _ in range(args.num_pyramid)
        ])

        self.tail_conv = nn.Sequential(
            common.ConvBNReLU2D(in_channels=args.num_features, out_channels=args.num_features, kernel_size=3, padding=1,
                                act=args.act),
            common.ConvBNReLU2D(in_channels=args.num_features, out_channels=args.in_channels, kernel_size=3, padding=1, act=args.act),
        )

        self.res_scale = common.Scale(0)

    def forward(self, samples):
        lr, rgb, lr_up = samples['img_lr'], samples['img_rgb'], samples['lr_up']
        lr_feature, guide_feature = self.head(lr_up, rgb)
        depth_features, guide_features = [lr_feature], [guide_feature]
        for num_p in range(self.num_pyramid - 1):
            lr_feature = self.depth_pyramid[num_p](lr_feature)
            depth_features.append(lr_feature)
            guide_feature = self.guide_pyramid[num_p](guide_feature)
            guide_features.append(guide_feature)
        # 从小到大
        depth_features, guide_features = list(reversed(depth_features)), list(reversed(guide_features))
        lr_input = None
        ret_feature = []
        for i in range(self.num_pyramid):
            h, w = depth_features[i].size()[2:]

            lr_input = self.init_conv[i](interpolate(lr, size=(h // 2, w // 2), mode='nearest'), lr_input)
            lr_input = self.up_sample[i](depth_features[i], guide_features[i], lr_input)

            ret_feature.append(lr_input)

        out = []

        out1, out2 = None, None
        for i in range(self.args.num_pyramid):
            out1, out2 = self.p_layers[i](ret_feature[i], out1)
            out2 = interpolate(out2, size=rgb.size()[2:], mode='bilinear', align_corners=False) + lr_up
            out.append(out2)

        return {'img_out': out[-1]}
