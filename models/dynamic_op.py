# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from models.common import sub_filter_start_end, get_same_padding


class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(self, max_in_channels, kernel_size_list=(3, 5, 7), stride=1, dilation=1, channels_per_group=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.channels_per_group = channels_per_group
        assert self.max_in_channels % self.channels_per_group == 0
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels // self.channels_per_group, bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, in_channels=None, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size

        if in_channels is None:
            in_channels = x.size(1)

        assert in_channels % self.channels_per_group == 0

        filters = self.get_active_filter(in_channels, kernel_size).contiguous()
        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            x[:, :in_channels, :, :], filters, None, self.stride, padding, self.dilation, in_channels // self.channels_per_group
        )
        return y


class DynamicPointConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicPointConv2d, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, stride=self.stride, bias=False)

    def forward(self, x, in_channel=None, out_channel=None):
        if in_channel is None:
            in_channel = x.size(1)
        if out_channel is None:
            out_channel = self.out_channels
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x[:, :in_channel, :, :], filters, None, self.stride, padding, self.dilation, 1)
        return y


# class DynamicBlock(nn.Module):
#     def __init__(self, in_channels, kernel_size_list=(3, 5, 7), stride=1, dilation=1):
#         super(DynamicBlock, self).__init__()
#         self.layers = nn.Sequential(
#             DynamicPointConv2d()
#         )

# net = DynamicPointConv2d(2, 2)
#
# arr = torch.randn(1, 2, 1, 1)
# brr = torch.randn(1, 1, 1, 1)
# crr = torch.randn(1, 1, 1, 1)
#
# loss = nn.MSELoss()
# opt = torch.optim.Adam(net.parameters(), lr=0.1)
#
# out = net(arr, 1, 1)
# print(out.size())
# error  = loss(brr, out)
# error.backward()
# opt.step()
# for name, parms in net.named_parameters():
#     print('Name', name, 'para', parms.reshape(1, -1), 'grad', parms.grad.reshape(1, -1))
#
# print('---')
# out = net(arr, 1)
# error  = loss(brr, out)
# error.backward()
# opt.step()
# for name, parms in net.named_parameters():
#     print('Name', name, 'para', parms.reshape(1, -1), 'grad', parms.grad.reshape(1, -1))
#
# print('---')
# out = net(arr, 1)
# error  = loss(brr, out)
# error.backward()
# opt.step()
# for name, parms in net.named_parameters():
#     print('Name', name, 'para', parms.reshape(1, -1), 'grad', parms.grad.reshape(1, -1))