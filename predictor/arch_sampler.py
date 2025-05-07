# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.nn.functional import interpolate
from utils import MetricLogger, SmoothedValue
from utils import to_device, metrics, normalize, de_normalize
from utils import get_random_architecture, key2arch, arch2key
from utils.base_lookup_table import get_model_efficiency
from predictor.bgru import Estimator

@torch.no_grad()
def get_arch_score(args, model, device, arch, val_data, end_epochs=2):
    model.eval()
    criterion = torch.nn.L1Loss()
    metric_logger = MetricLogger(delimiter="  ")

    metric_logger.add_meter("LOSS", SmoothedValue(window_size=1, fmt="{value}"))
    for _, samples in enumerate(val_data):
        samples['img_lr'], samples['lr_up'] = normalize(
            samples['img_lr'], samples['lr_up'], gt_img=samples['img_gt'], attr=args.dataset, args=args)
        if args.with_noisy:
            noise = torch.FloatTensor(samples['img_lr'].size()).normal_(mean=0, std=5 / 255.)
            samples['img_lr'] = torch.clamp(samples['img_lr'] + noise, 0, 1)
            samples['lr_up'] = interpolate(samples['img_lr'], scale_factor=args.scale, mode='bicubic', align_corners=False)
        with torch.no_grad():
            outputs = model(to_device(samples, device), arch)

        img_out = de_normalize(outputs['img_out'], gt_img=samples['img_gt'], attr=args.dataset, args=args)
        loss = criterion(img_out, samples['img_gt'])
        # rmse = metrics(
        #     img_out, samples['img_gt'], samples['img_mask'], args.gdata, attr='NYU', dataset=args.dataset)
        metric_logger.meters["LOSS"].update(loss, n=samples['img_gt'].size(0))
        if _ == end_epochs:
            break
    metric_logger.synchronize_between_processes()

    metric = {k: round(meter.global_avg, 8) for k, meter in metric_logger.meters.items()}
    model.train()

    return metric['LOSS']


def sample_arch(args, model, val_data, lookup_table, device, estimator):
    max_flops, min_flops = lookup_table.get_max_flops(), lookup_table.get_min_flops()
    flops_interval = (max_flops - min_flops) / 50

    # target_flops

    arch_list = []
    iter_times = 0
    arch_cand = get_random_architecture(args.num_stages, args.num_blocks, args.num_base_ops, args.num_fuse_ops)
    target_flops = get_model_efficiency(arch_cand, lookup_table.lookup_table, args)['FLOPs']

    target_flops = min(target_flops, max_flops - flops_interval)
    arch_list.append(key2arch(arch_cand, args.num_stages, args.num_blocks))

    my_pred_acc = []
    # print('--')
    if args.sampling in ['B', 'P']:
        while len(arch_list) < args.num_archs and iter_times < 500:
            arch_cand = get_random_architecture(args.num_stages, args.num_blocks, args.num_base_ops, args.num_fuse_ops)
            flops = get_model_efficiency(arch_cand, lookup_table.lookup_table, args)['FLOPs']
            if target_flops < flops < target_flops + flops_interval:
                arch_list.append(key2arch(arch_cand, args.num_stages, args.num_blocks))
            iter_times += 1

        if args.sampling == 'B':
            for arch in arch_list:
                my_pred_acc.append(get_arch_score(args, model, device, arch, val_data))
            return arch_list[np.argmin(my_pred_acc)]
        else:
            arch_tensor = []
            for arch in arch_list:
                arch_tensor.append(np.array(arch2key(arch, args.num_stages, args.num_blocks)).reshape(1, -1))

            arch_tensor = torch.LongTensor(np.concatenate(arch_tensor, axis=0)).cuda()
            if arch_tensor.size(0) == 1:
                arch_tensor = torch.cat((arch_tensor, arch_tensor), dim=0)
            out = estimator.predict(arch_tensor)
            return arch_list[torch.argmin(out).item()]


    return arch_list[0]
