# -*- coding: utf-8 -*-

import torch

import torch.nn as nn
import torch.nn.functional as F
from models.local_arch import Local_Base
from models.common import ConvBNReLU2D
from collections import OrderedDict


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        # x = x + inp
        # print(x.size(), samples['lr_up'].size())
        # print(x.shape)
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 4, 256, 256), base_size=(640, 640), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        if base_size is None:
            base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)



# def get_bare_model(self, net):
#     """Get bare model, especially under wrapping with
#     DistributedDataParallel or DataParallel.
#     """
#     if isinstance(net, (DataParallel, DistributedDataParallel)):
#             net = net.module
#         return net

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


class PBVS(nn.Module):
    def __init__(self, embed_dim):
        super(PBVS, self).__init__()
        self.head = nn.Sequential(
            ConvBNReLU2D(1, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act='PReLU', padding=1)
        )

    def forward(self, sample):
        return self.head(sample['lr_up'])


class NIR(nn.Module):
    def __init__(self, embed_dim, in_channels, guidance_channels):
        super(NIR, self).__init__()
        self.rgb = nn.Sequential(
            ConvBNReLU2D(guidance_channels, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act='PReLU', padding=1),
        )

        self.nir = nn.Sequential(
            ConvBNReLU2D(in_channels, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act='PReLU', padding=1),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU2D(embed_dim * 2 + in_channels + guidance_channels, out_channels=embed_dim, kernel_size=3, padding=1),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act='PReLU', padding=1),
        )

    def forward(self, sample):
        nir, rgb = sample['lr_up'], sample['img_rgb']
        # print(nir.shape, rgb.shape)
        out = torch.cat((self.rgb(rgb), self.nir(nir), rgb, nir), dim=1)
        return self.fuse(out)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        self.args = args
        embed_dim = 20
        enc_blks = [2, 2, 4, 6]
        middle_blk_num = 2
        dec_blks = [8, 4, 2, 2]

        self.head = NIR(embed_dim, in_channels=args.in_channels, guidance_channels=args.guidance_channels)
        train_size = (1, embed_dim, args.patch_size, args.patch_size)
        if args.test_only and args.tlc_enhance:
            self.net = NAFNetLocal(img_channel=embed_dim, width=embed_dim, middle_blk_num=middle_blk_num,
                                   enc_blk_nums=enc_blks, base_size=(448, 640),
                                   dec_blk_nums=dec_blks, train_size=train_size)
        else:
            self.net = NAFNet(img_channel=embed_dim, width=embed_dim, middle_blk_num=middle_blk_num,
                              enc_blk_nums=enc_blks,
                              dec_blk_nums=dec_blks)

        self.tail = ConvBNReLU2D(in_channels=embed_dim, out_channels=args.in_channels, kernel_size=3, padding=1)
        # if args.pre_trained:
        #     load_net = torch.load(args.load_name, map_location=lambda storage, loc: storage)
        #     self.net.load_state_dict(load_net['model'])
        #     print('Successfully load pre-trained model ...')

    def forward(self, samples):

        out = self.tail(self.net(self.head(samples)))
        return {'img_out': out + samples['lr_up']}

#  Base3 总多了decoder 网络的数量


def make_model(args): return Net(args)

# if __name__ == '__main__':
#     img_channel = 3
#     width = 32
#
#     # enc_blks = [2, 2, 4, 8]
#     # middle_blk_num = 12
#     # dec_blks = [2, 2, 2, 2]
#
#     enc_blks = [1, 1, 1, 28]
#     middle_blk_num = 1
#     dec_blks = [1, 1, 1, 1]
#
#     net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
#                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
#
#     inp_shape = (3, 256, 256)
#
#     from ptflops import get_model_complexity_info
#
#     macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
#
#     params = float(params[:-3])
#     macs = float(macs[:-4])
#
#     print(macs, params)