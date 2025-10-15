# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   base_lookup_table.py
@Time    :   2022/10/8 09:02
@Desc    :
"""
import os
import tqdm
import torch
import logging
import collections
import numpy as np
import pandas as pd
from models import mobile
from models import fusion
from models.common import ConvBNReLU2D
from models.fix_dcnas import StaticModel
from utils import get_random_architecture
from flop_counter.flop_count import FlopCountAnalysis
from utils.architectures import key2arch, get_fix_architecture, arch2key
from models.dcnas import DownSample, UpSample, FeatureInitialization, Tail

NUM_ITER = 50
h, w = 480, 480


@torch.no_grad()
def calculate_latency(net, b, cd, cg, h, w, device):
    dep = torch.randn(b, cd, h, w).float().to(device)
    rgb = torch.randn(b, cg, h, w).float().to(device)
    net.float().to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(NUM_ITER):
        net(None, None, dep, rgb)

    end.record()
    torch.cuda.synchronize()
    end_time = start.elapsed_time(end)
    torch.cuda.empty_cache()
    return end_time / NUM_ITER


@torch.no_grad()
def calculate_head_latency(net, b, cd, cg, h, w, device):
    dep = torch.randn(b, cd, h, w).float().to(device)
    rgb = torch.randn(b, cg, h, w).float().to(device)
    net.float().to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_ITER):
        net(dep, rgb)

    end.record()
    torch.cuda.synchronize()
    end_time = start.elapsed_time(end)
    torch.cuda.empty_cache()
    return end_time / NUM_ITER


@torch.no_grad()
def single_latency(net, b, c, h, w, device):
    x = torch.randn(b, c, h, w).float().to(device)
    net.float().to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(NUM_ITER):
        net(x)

    end.record()
    torch.cuda.synchronize()
    end_time = start.elapsed_time(end)
    torch.cuda.empty_cache()
    return end_time / NUM_ITER


@torch.no_grad()
def double_latency(net, b, c, h, w, device):
    dep = torch.randn(b, c, h, w).float().to(device)
    rgb = torch.randn(b, c, h, w).float().to(device)
    net.cuda().to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_ITER):
        net(dep, rgb)
    end.record()
    torch.cuda.synchronize()
    end_time = start.elapsed_time(end)
    torch.cuda.empty_cache()
    return end_time / NUM_ITER


@torch.no_grad()
def calculate_parameters(net, h, w):
    return sum(p.numel() for p in net.parameters()) / 1e6
    # return sum(p.numel() for p in net.parameters() if p.requires_grad)


@torch.no_grad()
def calculate_flops(net, cd, cg, h, w, device):
    dep = torch.randn(1, cd, h, w).float().to(device)
    rgb = torch.randn(1, cg, h, w).float().to(device)
    flops = FlopCountAnalysis(net.float().to(device), (None, None, dep, rgb))
    return flops.total() / 1e9


@torch.no_grad()
def calculate_head_flops(net, cd, cg, h, w, device):
    dep = torch.randn(1, cd, h, w).float().to(device)
    rgb = torch.randn(1, cg, h, w).float().to(device)
    flops = FlopCountAnalysis(net.float().to(device), (dep, rgb))
    return flops.total() / 1e9


@torch.no_grad()
def calculate_model_efficiency(net, b, cd, cg, h, w, device):
    model_efficiency = collections.OrderedDict()
    model_efficiency['Latency'] = calculate_latency(net, b, cd, cg, h, w, device)
    model_efficiency['FLOPs'] = calculate_flops(net, cd, cg, h, w, device)
    model_efficiency['Paras'] = calculate_parameters(net, h, w)
    return model_efficiency


@torch.no_grad()
def single_flops(net, c, h, w, device):
    x = torch.randn(1, c, h, w).float().to(device)
    flops = FlopCountAnalysis(net.float().to(device), (x, ))
    return flops.total() / 1e9


@torch.no_grad()
def double_flops(net, c,  h, w, device):
    dep = torch.randn(1, c, h, w).float().to(device)
    rgb = torch.randn(1, c, h, w).float().to(device)
    flops = FlopCountAnalysis(net.float().to(device), (dep, rgb))
    return flops.total() / 1e9


@torch.no_grad()
def generate_lookup_table(file_path, args, override=False):
    if os.path.exists(file_path) and not override:
        print('Reading Lookup Table form {}'.format(file_path))
        return pd.read_csv(file_path, index_col=[0])
    else:
        print('===> Generating Lookup Table')
    # print('Generating Lookup Table ...')
    input_features = [args.num_features * pow(2, i) for i in range(args.num_stages)]
    input_resolutions = [(h // pow(2, i), w // (pow(2, i))) for i in range(args.num_stages)]
    # print(input_resolutions, input_features)

    lookup_table = {}
    for num_features in input_features:
        for resolution in input_resolutions:
            # Head
            key = f'head_{num_features}_{resolution[0]}_{resolution[1]}'
            # num_features, in_channels, guidance_channels, act, bias, norm
            # model = FeatureInitialization(num_features, args.act, args.bias, args.norm)
            model = FeatureInitialization(
                num_features=args.num_features, in_channels=args.in_channels,
                guidance_channels=args.guidance_channels, act=args.act, bias=args.bias, norm=args.norm
            )
            paras = calculate_parameters(model, 0, 0)
            flops = calculate_head_flops(model, args.in_channels, args.guidance_channels, resolution[0], resolution[1], args.device)
            latency = calculate_head_latency(model, args.batch_size, args.in_channels, args.guidance_channels, resolution[0], resolution[1], args.device)
            lookup_table[key] = {'flops': flops, 'paras': paras, 'latency': latency}
            # Middle

            key = f'middle_{num_features}_{resolution[0]}_{resolution[1]}'
            model = ConvBNReLU2D(num_features, num_features, 1, bias=args.bias, act=args.act)
            paras = calculate_parameters(model, 0, 0)
            flops = single_flops(model, num_features, resolution[0], resolution[1], args.device)
            latency = single_latency(model, args.batch_size, num_features, resolution[0], resolution[1], args.device)
            lookup_table[key] = {'flops': flops, 'paras': paras, 'latency': latency}

            # Tail
            key = f'tail_{num_features}_{resolution[0]}_{resolution[1]}'
            model = Tail(num_features, act=args.act, out_channels=args.in_channels, bias=args.bias)
            paras = calculate_parameters(model, 0, 0)
            flops = single_flops(model, num_features, resolution[0], resolution[1], args.device)
            latency = single_latency(model, args.batch_size, num_features, resolution[0], resolution[1], args.device)
            lookup_table[key] = {'flops': flops, 'paras': paras, 'latency': latency}

            # Down  and Up
            key = f'up_{num_features}_{resolution[0]}_{resolution[1]}'
            model = UpSample(scale=2, num_features=num_features, bias=args.bias, keep_dim=False)
            paras = calculate_parameters(model, 0, 0)
            flops = single_flops(model, num_features, resolution[0], resolution[1], args.device)
            latency = single_latency(model, args.batch_size, num_features, resolution[0], resolution[1], args.device)
            lookup_table[key] = {'flops': flops, 'paras': paras, 'latency': latency}

            key = f'down_{num_features}_{resolution[0]}_{resolution[1]}'
            model = DownSample(scale=2, num_features=num_features, bias=args.bias, keep_dim=False)
            paras = calculate_parameters(model, 0, 0)
            flops = double_flops(model, num_features, resolution[0], resolution[1], args.device)
            latency = double_latency(model, args.batch_size, num_features, resolution[0], resolution[1], args.device)
            lookup_table[key] = {'flops': flops, 'paras': paras, 'latency': latency}

            # Base
            for base_key in mobile.block_keys:
                model = mobile.block_dict[base_key](num_features, num_features, 1, args.act, bias=False, norm=args.norm)
                key = f'{base_key}_{num_features}_{resolution[0]}_{resolution[1]}'
                if base_key.find('skip') != -1:
                    paras, flops, latency = 0, 0, 0
                else:
                    paras = calculate_parameters(model, 0, 0)
                    flops = single_flops(model, num_features, resolution[0], resolution[1], args.device)
                    latency = single_latency(model, args.batch_size, num_features, resolution[0], resolution[1], args.device)
                lookup_table[key] = {'flops': flops, 'paras': paras, 'latency': latency}

            for fuse_key in fusion.block_keys:
                model = fusion.block_dict[fuse_key](num_features)
                key = f'{fuse_key}_{num_features}_{resolution[0]}_{resolution[1]}'
                if fuse_key.find('skip') != -1:
                    paras, flops, latency = 0, 0, 0
                else:
                    paras = calculate_parameters(model, 0, 0)
                    flops = double_flops(model, num_features, resolution[0], resolution[1], args.device)
                    latency = double_latency(model, max(1, args.batch_size // 2), num_features, resolution[0], resolution[1], args.device)
                lookup_table[key] = {'flops': flops, 'paras': paras, 'latency': latency}

    lookup_table = pd.DataFrame(lookup_table).T

    lookup_table.to_csv(file_path)
    return lookup_table


@torch.no_grad()
def get_model_efficiency(cand_key, lookup_table, args):
    # print(arch)
    total_flops = []
    total_paras = []
    total_latency = []

    num_features = [args.num_features * pow(2, i) for i in range(args.num_stages)]
    input_resolutions = [(h // pow(2, i), w // (pow(2, i))) for i in range(args.num_stages)]
    # print(input_resolutions)
    # print(num_features)

    arch = key2arch(cand_key, args.num_stages, args.num_blocks) if isinstance(cand_key, tuple) else cand_key
    fuse_ops = arch['fuse_op']
    num_skip = 0
    for idx in list(fuse_ops):
        if fusion.block_keys[idx] in ['skip', 'skip_rgb']:
            num_skip += 1
        else:
            break
    # print(num_skip)
    # head
    head_key = f'head_{args.num_features}_{h}_{w}'
    num_div = 2 if num_skip == args.num_stages else 1
    total_flops.append(lookup_table.loc[head_key, 'flops'] / num_div)
    total_paras.append(lookup_table.loc[head_key, 'paras'] / num_div)
    total_latency.append(lookup_table.loc[head_key, 'latency'] / num_div)

    # print(head_key, lookup_table.loc[head_key, 'paras'])
    # Middle
    middle_key = f'middle_{num_features[-1]}_{input_resolutions[-1][0]}_{input_resolutions[-1][1]}'
    total_flops.append(lookup_table.loc[middle_key, 'flops'])
    total_paras.append(lookup_table.loc[middle_key, 'paras'])
    total_latency.append(lookup_table.loc[middle_key, 'latency'])

    # print(middle_key, lookup_table.loc[middle_key, 'paras'])

    # Tail
    tail_key = f'tail_{num_features[0]}_{h}_{w}'
    total_flops.append(lookup_table.loc[tail_key, 'flops'])
    total_paras.append(lookup_table.loc[tail_key, 'paras'])
    total_latency.append(lookup_table.loc[tail_key, 'latency'])

    # print(tail_key, lookup_table.loc[tail_key, 'paras'])

    # Fuse layer
    for i in range(args.num_stages):
        model_key = fusion.block_keys[arch['fuse_op'][i]]
        i_h = input_resolutions[args.num_stages - i - 1][0]
        i_w = input_resolutions[args.num_stages - i - 1][1]
        base_key = f'{model_key}_{num_features[args.num_stages - i - 1]}_{i_h}_{i_w}'
        total_flops.append(lookup_table.loc[base_key, 'flops'])
        total_paras.append(lookup_table.loc[base_key, 'paras'])
        total_latency.append(lookup_table.loc[base_key, 'latency'])

        # print(base_key, lookup_table.loc[base_key, 'paras'])

    # Up and Down Sample
    for i in range(args.num_stages - 1):
        i_h = input_resolutions[args.num_stages - i - 1][0]
        i_w = input_resolutions[args.num_stages - i - 1][1]
        base_key = f'up_{num_features[args.num_stages - i - 1]}_{i_h}_{i_w}'
        total_flops.append(lookup_table.loc[base_key, 'flops'])
        total_paras.append(lookup_table.loc[base_key, 'paras'])
        total_latency.append(lookup_table.loc[base_key, 'latency'])

        # print(base_key, lookup_table.loc[base_key, 'paras'])

    for i in range(args.num_stages - 1):
        i_h = input_resolutions[i][0]
        i_w = input_resolutions[i][1]
        base_key = f'down_{num_features[i]}_{i_h}_{i_w}'
        if i < args.num_stages - 1 - num_skip:
            total_flops.append(lookup_table.loc[base_key, 'flops'])
            total_paras.append(lookup_table.loc[base_key, 'paras'])
            total_latency.append(lookup_table.loc[base_key, 'latency'])

        else:
            total_flops.append(lookup_table.loc[base_key, 'flops'] / 2)
            total_paras.append(lookup_table.loc[base_key, 'paras'] / 2)
            total_latency.append(lookup_table.loc[base_key, 'latency'] / 2)

            # print(base_key, lookup_table.loc[base_key, 'paras'] / 2)

    for name in ['rgb_encoder', 'dep_encoder', 'rec_decoder']:
        for i in range(args.num_stages - num_skip if name == 'rgb_encoder' else args.num_stages):
            for j in range(args.num_blocks):
                model_key = mobile.block_keys[arch[name][i][j]]
                if name == 'rec_decoder':
                    in_features = num_features[args.num_stages - 1 - i]
                    i_h = input_resolutions[args.num_stages - 1 - i][0]
                    i_w = input_resolutions[args.num_stages - 1 - i][1]
                else:
                    in_features = num_features[i]
                    i_h, i_w = input_resolutions[i][0], input_resolutions[i][1]
                base_key = f'{model_key}_{in_features}_{i_h}_{i_w}'

                total_flops.append(lookup_table.loc[base_key, 'flops'])
                total_paras.append(lookup_table.loc[base_key, 'paras'])
                total_latency.append(lookup_table.loc[base_key, 'latency'])
                # print(base_key, lookup_table.loc[base_key, 'flops'])

    return {'Latency': np.sum(total_latency), 'FLOPs': np.sum(total_flops), 'Paras': np.sum(total_paras)}


class LookUpTable(object):
    def __init__(self, args, file_path):

        self.lookup_table = pd.read_csv(file_path, header=0, index_col=0)
        self.model_statics = pd.read_csv(file_path.replace('.csv', '_static.csv'), header=0, index_col=0)

    def get_max_flops(self):
        return self.model_statics['FLOPs'].max()

    def get_min_flops(self):
        return self.model_statics['FLOPs'].min()

    def get_max_paras(self):
        return self.model_statics['Paras'].max()

    def get_min_paras(self):
        return self.model_statics['Paras'].min()

    def get_max_latency(self):
        return self.model_statics['Latency'].max()

    def get_min_latency(self):
        return self.model_statics['Latency'].min()















