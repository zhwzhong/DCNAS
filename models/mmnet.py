# -*- coding: utf-8 -*-


import torch
import warnings
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F



class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.kernel_size = 3
        self.stride = 2
        self.padding = 1

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g.weight)
        nn.init.constant(self.g.bias, 0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W[0].weight)
            nn.init.constant(self.W[0].bias, 0)
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)

            self.W_pan = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W_pan[0].weight)
            nn.init.constant(self.W_pan[0].bias, 0)
            nn.init.constant(self.W_pan[1].weight, 0)
            nn.init.constant(self.W_pan[1].bias, 0)

        else:
            self.W_pan = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                                 kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W_pan.weight)
            nn.init.constant(self.W_pan.bias, 0)

            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.phi_pan = None

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.phi_pan = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        ## PAN
        self.g_pan = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g_pan.weight)
        nn.init.constant(self.g_pan.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            self.g_pan = nn.Sequential(self.g_pan, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
                self.phi_pan = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
                self.phi_pan = nn.Sequential(self.phi_pan, max_pool(kernel_size=2))

    def forward(self, x, x_pan):
        output = self._embedded_gaussian(x, x_pan)
        return output

    def _embedded_gaussian(self, x, x_pan):
        batch_size = x.size(0)
        patch_size = x.size(2)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h/scale, w/scale)]->(b, thw/scale^2, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h/scale, w/scale)]->(b, 0.5c, thw/scale^2)
        # f=>(b, thw/scale^2, 0.5c)dot(b, 0.5c, thw/scale^2) = (b, thw/scale^2, thw/scale^2)
        theta_x = self.theta(x)
        theta_x = self.up(theta_x)
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x)
        phi_x = self.up(phi_x)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, int(patch_size), int(patch_size))
        W_y = self.W(y)

        # PAN
        x1 = x_pan
        g_pan_x = self.g_pan(x1).view(batch_size, self.inter_channels, -1)
        g_pan_x = g_pan_x.permute(0, 2, 1)

        phi_pan_x = self.phi_pan(x1)
        phi_pan_x = self.up(phi_pan_x)
        phi_pan_x = phi_pan_x.view(batch_size, self.inter_channels, -1)

        f_pan = torch.matmul(theta_x, phi_pan_x)
        f_pan_div_C = F.softmax(f_pan, dim=-1)
        y_pan = torch.matmul(f_pan_div_C, g_pan_x)
        y_pan = y_pan.permute(0, 2, 1).contiguous()
        y_pan = y_pan.view(batch_size, self.inter_channels, int(patch_size), int(patch_size))
        W_pan_y = self.W_pan(y_pan)

        z = torch.cat([W_y, W_pan_y], 1)

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)

warnings.filterwarnings('ignore')

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, pad_model=None, groups=1):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        if self.pad_model == None:
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding,
                                        groups=groups, bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,
                                        groups=groups, bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu',
                 norm='batch', pad_model=None, groups=1):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale

        if self.norm == 'batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias, groups=groups)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias, groups=groups)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None,
                        [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer,
                         self.act])  #
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class C_RB(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1,
                 bias=False, activation='prelu', norm=None, n_resblocks=2):
        super(C_RB, self).__init__()
        res_block = [
            ConvBlock(input_channel, output_channel, kernel_size, stride, padding, activation=activation,
                      norm=norm, bias=bias),
        ]
        for i in range(n_resblocks):
            res_block.append(
                ResnetBlock(output_channel, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm))
        # res_block.append(ConvBlock(output_channel, output_channel, kernel_size, stride, padding, activation=activation,norm=norm, bias=bias))
        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return self.res_block(x)


class RB(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1,
                 bias=False, activation='prelu', norm=None, n_resblocks=2):
        super(RB, self).__init__()
        res_block = []
        for i in range(n_resblocks):
            res_block.append(
                ResnetBlock(output_channel, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm))
        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return self.res_block(x)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class U_block(nn.Module):
    def __init__(self, in_channel=4, mid_channel=4):
        super(U_block, self).__init__()
        self.conv1 = ConvBlock(in_channel + 4, mid_channel, 5, 1, 2, activation='prelu', norm=None, bias=False)
        self.conv2 = RB(mid_channel, in_channel, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.panhead = ConvBlock(1, 3, 5, 1, 2, activation='prelu', norm=None, bias=False)

    def forward(self, input, pan):
        pan_f = self.panhead(pan)
        pan_f = torch.cat((pan_f, pan), dim=1)
        x = torch.cat((input, pan_f), dim=1)
        x = self.conv1(x)
        U = self.conv2(x)
        mem_f = torch.cat((U, x, input), dim=1)
        return U, mem_f


class Hl_block(nn.Module):
    def __init__(self, in_channel=4, mid_channel=4, o_channel=4):
        super(Hl_block, self).__init__()
        self.downconv = ConvBlock(in_channel, mid_channel, 5, 4, 2, activation='prelu', norm=None, bias=False)
        self.upconv = nn.ConvTranspose2d(mid_channel, o_channel, 5, 4, 1, 1, bias=False)
        self.act = nn.PReLU(init=0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, H, U, L, mem):
        if mem is not None:
            H_f = torch.cat((H, mem), dim=1)
        else:
            H_f = H
        DKH = self.downconv(H_f)
        H = H - (self.alpha * self.act(self.upconv(L - DKH)) + self.beta * (H - U))
        mem_f = torch.cat((H_f, H), dim=1)
        return H, mem_f


class V_block(nn.Module):
    def __init__(self, in_channel=4, mid_channel=4):
        super(V_block, self).__init__()
        self.conv1 = ConvBlock(in_channel + 4, mid_channel, 5, 1, 2, activation='prelu', norm=None, bias=False)
        self.conv2 = RB(mid_channel, in_channel, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.panhead = ConvBlock(1, 3, 5, 1, 2, activation='prelu', norm=None, bias=False)

    def forward(self, input, pan):
        pan_f = self.panhead(pan)
        pan_f = torch.cat((pan_f, pan), dim=1)
        x = torch.cat((input, pan_f), dim=1)
        x = self.conv1(x)
        V = self.conv2(x)
        mem_f = torch.cat((V, x, input), dim=1)
        return V, mem_f


class Hp_block(nn.Module):
    def __init__(self, in_channel=4, mid_channel=4, o_channel=4):
        super(Hp_block, self).__init__()
        self.downconv = ConvBlock(in_channel, mid_channel, 5, 4, 2, activation='prelu', norm=None, bias=False)
        self.upconv = nn.ConvTranspose2d(mid_channel, o_channel, 5, 4, 1, 1,
                                         bias=False)  # ConvTranspose2d (in-1)*stride+outpad-2*pad+kernelsize
        self.act = nn.PReLU(init=0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, SH, V, L, mem):
        if mem is not None:
            H_f = torch.cat((SH, mem), dim=1)
        else:
            H_f = SH
        DKSH = self.downconv(H_f)
        SH = SH - (self.alpha * self.act(self.upconv(L - DKSH)) + self.beta * (SH - V))
        mem_f = torch.cat((H_f, SH), dim=1)
        return SH, mem_f


class vanilla_stage(nn.Module):
    def __init__(self):
        super(vanilla_stage, self).__init__()
        self.U = U_block(in_channel=4, mid_channel=4)
        self.Hl = Hl_block(in_channel=4, mid_channel=4)
        self.V = V_block(in_channel=4, mid_channel=4)
        self.Hp = Hp_block(in_channel=4, mid_channel=4)
        # self.H2SH=ConvBlock(4,4,3, 1, 1, activation='prelu', norm=None, bias = False)
        # self.H2SH = C_RB(4, 4, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.H2SH_Hin = ConvBlock(4, 4, 3, 2, 1, activation='prelu', norm=None, bias=False)
        self.H2SH_PANin = ConvBlock(1, 4, 3, 2, 1, activation='prelu', norm=None, bias=False)
        self.H2SH = NONLocalBlock2D(in_channels=4, mode='embedded_gaussian')
        self.H2SH_out = C_RB(8, 4, 1, 1, 0, bias=False, activation='prelu', norm=None, n_resblocks=1)
        self.H2SH_outup = nn.ConvTranspose2d(4, 4, 3, 2, 1, 1,
                                             bias=False)  # ConvTranspose2d (in-1)*stride+outpad-2*pad+kernelsize

    def forward(self, H, L, pan):
        U, U_mem_out = self.U(H, pan)
        H, Hl_mem_out = self.Hl(H, U, L, None)

        SH = self.H2SH_Hin(H)
        pan_in = self.H2SH_PANin(pan)
        SH = self.H2SH(SH, pan_in)
        SH = self.H2SH_out(SH)
        SH = self.H2SH_outup(SH)

        V, V_mem_out = self.V(SH, pan)
        SH, Hp_mem_out = self.Hp(SH, V, L, None)
        return SH, U_mem_out, Hl_mem_out, V_mem_out, Hp_mem_out


class Net(nn.Module):
    def __init__(self, base_filter=None, args=None, num_channels=4, stage=3, h=128, w=128):
        super(Net, self).__init__()
        self.stage_num = stage
        self.stage0 = vanilla_stage()
        ## memory flow
        self.Uconv1 = C_RB(12, 8, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.cellU = ConvLSTMCell((h, w), 8, 8, [3, 3], False)
        self.Uconv2 = C_RB(8, 8, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)

        self.Hlconv1 = C_RB(12, 8, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.cellHl = ConvLSTMCell((h, w), 8, 8, [3, 3], False)
        self.Hlconv2 = C_RB(8, 4, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)

        self.Vconv1 = C_RB(12, 8, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.cellV = ConvLSTMCell((h, w), 8, 8, [3, 3], False)
        self.Vconv2 = C_RB(8, 8, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)

        self.Hpconv1 = C_RB(12, 8, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.cellHp = ConvLSTMCell((h, w), 8, 8, [3, 3], False)
        self.Hpconv2 = C_RB(8, 4, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)

        ## information flow
        self.Uconv3 = C_RB(12, 4, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)
        self.H2SH_Hin = ConvBlock(4, 4, 3, 2, 1, activation='prelu', norm=None, bias=False)
        self.H2SH_PANin = ConvBlock(1, 4, 3, 2, 1, activation='prelu', norm=None, bias=False)
        self.H2SH = NONLocalBlock2D(in_channels=4, mode='embedded_gaussian')
        self.H2SH_out = C_RB(8, 4, 1, 1, 0, bias=False, activation='prelu', norm=None, n_resblocks=1)
        self.H2SH_outup = nn.ConvTranspose2d(4, 4, 3, 2, 1, 1,
                                             bias=False)  # ConvTranspose2d (in-1)*stride+outpad-2*pad+kernelsize
        self.Vconv3 = C_RB(12, 4, 3, 1, 1, bias=False, activation='prelu', norm=None, n_resblocks=2)

        self.U_block = U_block(in_channel=num_channels, mid_channel=num_channels)
        self.Hl_block = Hl_block(in_channel=num_channels * 2, mid_channel=num_channels, o_channel=num_channels)
        self.V_block = V_block(in_channel=num_channels, mid_channel=num_channels)
        self.Hp_block = Hp_block(in_channel=num_channels * 2, mid_channel=num_channels, o_channel=num_channels)

    def forward(self, samples):
        l_ms, bms, pan = samples['img_lr'], samples['lr_up'], samples['img_rgb']
        H = bms

        u_h, u_c = self.cellU.init_hidden(batch_size=pan.size(0))
        Hl_h, Hl_c = self.cellHl.init_hidden(batch_size=pan.size(0))
        v_h, v_c = self.cellV.init_hidden(batch_size=pan.size(0))
        Hp_h, Hp_c = self.cellHp.init_hidden(batch_size=pan.size(0))

        SH, U_mem_f, Hl_mem_f, V_mem_f, Hp_mem_f = self.stage0(H, l_ms, pan)

        for i in range(self.stage_num):
            ###memory###
            if i != 0:
                Hl_mem_f = self.Hlconv1(Hl_mem_f)
                Hp_mem_f = self.Hpconv1(Hp_mem_f)
            U_mem_f = self.Uconv1(U_mem_f)
            V_mem_f = self.Vconv1(V_mem_f)

            u_h, u_c = self.cellU(U_mem_f, cur_state=[u_h, u_c])
            Hl_h, Hl_c = self.cellHl(Hl_mem_f, cur_state=[Hl_h, Hl_c])
            v_h, v_c = self.cellV(V_mem_f, cur_state=[v_h, v_c])
            Hp_h, Hp_c = self.cellHp(Hp_mem_f, cur_state=[Hp_h, Hp_c])

            U_mem_f = self.Uconv2(u_h)
            Hl_mem_f = self.Hlconv2(Hl_h)
            V_mem_f = self.Vconv2(v_h)
            Hp_mem_f = self.Hpconv2(Hp_h)

            ###main stream###
            H_f = torch.cat((SH, U_mem_f), dim=1)
            H_f = self.Uconv3(H_f)
            U, U_mem_f = self.U_block(H_f, pan)

            H, Hl_mem_f = self.Hl_block(SH, U, l_ms, Hl_mem_f)

            SH = self.H2SH_Hin(H)
            pan_in = self.H2SH_PANin(pan)
            SH = self.H2SH(SH, pan_in)
            SH = self.H2SH_out(SH)
            SH = self.H2SH_outup(SH) + H

            H_f = torch.cat((SH, V_mem_f), dim=1)
            H_f = self.Vconv3(H_f)
            V, V_mem_f = self.V_block(H_f, pan)

            SH, Hp_mem_f = self.Hp_block(SH, V, l_ms, Hp_mem_f)

        return {'img_out': SH}


def make_model(args): return Net(num_channels=args.in_channels)


if __name__ == '__main__':
    from config import args
    from calc_complexity import calculate_latency, calculate_parameters, calculate_flops

    args.in_channels = 4
    args.batch_size = 1
    cd, cg, h, w = 4, 1, 480, 480
    model = Net(num_channels=args.in_channels, h=h, w=w).cuda()
    samples = {
        'lr_up': torch.randn(1, cd, h, w).float().cuda(),
        'img_rgb': torch.randn(1, cg, h, w).float().cuda(),
        'lr_mask': torch.randn(1, cd, h // args.scale, w // args.scale).float().cuda(),
        'img_lr': torch.randn(1, cd, h // args.scale, w // args.scale).float().cuda()
    }
    print('Paras', calculate_parameters(model, h, w))
    print('FLOPs', calculate_flops(model, cd, cg, h, w, torch.device('cuda')))
    print('Latency', calculate_latency(model, cd, cg, h, w, torch.device('cuda')))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()

        for _ in range(10):
            model(samples)

        end.record()
        torch.cuda.synchronize()
        end_time = start.elapsed_time(end)
    print('Latency', end_time / 10)

