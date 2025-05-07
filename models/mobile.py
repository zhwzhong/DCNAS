# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   mobile.py
@Time    :   2022/10/7 21:12
@Desc    :
"""
import re
from torch import nn
from models.common import ConvBNReLU2D, get_act, CALayer
from models.dynamic_op import DynamicPointConv2d, DynamicSeparableConv2d


def extract_numbers(text):
    pattern = r'\d+'  # 正则表达式模式，匹配连续的数字
    numbers = re.findall(pattern, text)  # 使用 re.findall 提取所有匹配的数字
    return [int(i) for i in numbers]


block_dict = {
    'skip':
        lambda inp, oup, stride, act, bias, norm: nn.Identity(inp, oup, 3, 1, stride, 3, act, bias, norm),
	'mobilenet_3x3_ratio_2':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 3, 1, stride, 2, act, bias, norm),
    'mobilenet_3x3_ratio_3':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 3, 1, stride, 3, act, bias, norm),
    'mobilenet_3x3_ratio_4':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 3, 1, stride, 4, act, bias, norm),
    'mobilenet_3x3_ratio_5':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 3, 1, stride, 5, act, bias, norm),
    'mobilenet_3x3_ratio_6':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 3, 1, stride, 6, act, bias, norm),

    'mobilenet_5x5_ratio_2':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 5, 2, stride, 2, act, bias, norm),
    'mobilenet_5x5_ratio_3':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 5, 2, stride, 3, act, bias, norm),
    'mobilenet_5x5_ratio_4':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 5, 2, stride, 4, act, bias, norm),
    'mobilenet_5x5_ratio_5':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 5, 2, stride, 5, act, bias, norm),
    'mobilenet_5x5_ratio_6':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 5, 2, stride, 6, act, bias, norm),

	'mobilenet_7x7_ratio_2':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 7, 3, stride, 2, act, bias, norm),
    'mobilenet_7x7_ratio_3':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 7, 3, stride, 3, act, bias, norm),
    'mobilenet_7x7_ratio_4':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 7, 3, stride, 4, act, bias, norm),
    'mobilenet_7x7_ratio_5':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 7, 3, stride, 5, act, bias, norm),
    'mobilenet_7x7_ratio_6':
        lambda inp, oup, stride, act, bias, norm: InvertedResidual(inp, oup, 7, 3, stride, 6, act, bias, norm),
}

MAX_EXP_RATIO = 6
KERNEL_LIST = (3, 5, 7)
block_keys = list(block_dict.keys())


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, expand_ratio, act, bias=False, norm=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.layers = nn.Sequential(
            ConvBNReLU2D(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=bias, act=act, norm=norm),
            ConvBNReLU2D(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size, stride, padding,
                         groups=in_channels * expand_ratio, bias=bias, act=act, norm=norm),
            ConvBNReLU2D(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=bias, act=None, norm=norm)
        )
        self.SE = CALayer(out_channels)

    def forward(self, x):
        return x + self.SE(self.layers(x)) if self.use_res_connect else self.layers(x)

        # return x + self.layers(x) if self.use_res_connect else self.layers(x)


def get_para(choice):
    key = extract_numbers(block_keys[choice])
    return key[0], key[2]


class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act):
        super(BaseBlock, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.layer1 = DynamicPointConv2d(in_channels, in_channels * MAX_EXP_RATIO)
        self.act1 = get_act(act)
        self.layer2 = DynamicSeparableConv2d(in_channels * MAX_EXP_RATIO, kernel_size_list=KERNEL_LIST)
        self.act2 = get_act(act)
        self.layer3 = DynamicPointConv2d(in_channels * MAX_EXP_RATIO, out_channels)

    def forward(self, inputs, in_channels, exp_ratio, kernel_size):
        out = self.act1(self.layer1(inputs, in_channels, in_channels * exp_ratio))
        out = self.act2(self.layer2(out, in_channels * exp_ratio, kernel_size))
        out = self.layer3(out, in_channels * exp_ratio, in_channels)
        return inputs + out


class MBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act, bias, norm, choice=None):
        super(MBlock, self).__init__()

        self.fix_arch = False

        self.in_channels = in_channels
        self.num_operators = len(block_dict)

        # print(in_channels, out_channels, stride, act, bias, norm, choice)
        if choice is not None:
            self.fix_arch = True
            self.mb_layers = block_dict[block_keys[choice]](in_channels, out_channels, stride, act, bias, norm)
        else:
            self.mb_layers = BaseBlock(in_channels, out_channels, stride, act)

    def forward(self, x, choice=None):   # choice == 0 Identity
        assert choice is not None or self.fix_arch, 'Please Select One Op...'
        if choice is None and self.fix_arch:
            return self.mb_layers(x)
        elif choice == 0:
            return x
        else:
            kernel, exp_ratio = get_para(choice)
            return self.mb_layers(x, self.in_channels, exp_ratio, kernel)
