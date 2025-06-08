# -*- coding: utf-8 -*-

import tqdm
import torch
import numpy as np
from tabulate import tabulate
from logger import create_dir
from predictor.arch_sampler import sample_arch
from torch.nn.functional import interpolate
from utils import MetricLogger, SmoothedValue, model_parameters
from utils import mix_up, to_device, self_ensemble, normalize, de_normalize
from utils import dispatch_clip_grad, get_random_architecture, key2arch


def train_one_epoch(model, criterion, train_data, eval_data, lookup_table, optimizer, epoch,  args,  loss_scaler=None, estimator=None):
    model.train()
    device = torch.device(args.device)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    header = f"Epoch: [{epoch}]"
    num_steps = 0
    arch = None

    for samples in metric_logger.log_every(train_data, args.print_freq, header):
        # 计算均值和方差所用的图像
        samples = to_device(samples, device)
        norm_img = samples['img_lr'] if args.dataset.lower() in ['fastmri', 'm4raw'] else samples['img_gt']
        samples['img_lr'], samples['lr_up'] = normalize(
            samples['img_lr'], samples['lr_up'], gt_img=norm_img, attr=args.dataset, args=args)

        if args.with_noisy:
            noise = torch.FloatTensor(samples['img_lr'].size()).normal_(mean=0, std=5 / 255.).to(device)
            samples['img_lr'] = torch.clamp(samples['img_lr'] + noise, 0, 1)
            samples['lr_up'] = interpolate(samples['img_lr'], scale_factor=args.scale, mode='bicubic', align_corners=False)

        samples = mix_up(samples, args.alpha) if args.mix_up else samples

        if args.train_supernet:
            if num_steps % args.sample_interval == 0:
                arch = sample_arch(args, model, eval_data, lookup_table, device, estimator)
            outputs = model(samples, arch)
        else:
            outputs = model(samples, train=True) if args.model == 'GAD' else model(samples)
        outputs['img_out'] = de_normalize(outputs['img_out'], gt_img=norm_img, attr=args.dataset, args=args)
        loss = criterion(outputs['img_out'], samples['img_gt'], mask=samples['img_mask'])
        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    num_steps += 1
    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, val_data, test_name, args, arch=None, logger=None):

    model.eval()
    all_rmse = []
    # header = '{}:'.format(test_name)
    device = torch.device(args.device)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("RMSE/{}".format(test_name), SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("Time/{}".format(test_name), SmoothedValue(window_size=1, fmt="{value}"))

    sv_path = f'./results/{args.dataset}/{args.down_type}/{args.model}/{args.scale}/{test_name}/'
    if args.save_result:
        create_dir(sv_path)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    _test_name = args.dataset if test_name in ['val', 'test'] else test_name
    # for samples in metric_logger.log_every(val_data, len(val_data) + 1, header):

    nb = len(val_data)
    val_data = enumerate(val_data)
    if args.local_rank in [-1, 0] and not args.test_only:
        val_data = tqdm.tqdm(val_data, total=nb)  # 只在主进程打印进度条

    for _, samples in val_data:
        # 计算均值和方差所用的图像
        if args.dataset.lower() in ['fastmri', 'm4raw']: # channel 转为 batch
            samples['img_lr'] = samples['img_lr'].permute(1, 0, 2, 3).contiguous()
            samples['img_gt'] = samples['img_gt'].permute(1, 0, 2, 3).contiguous()
            samples['lr_up'] = samples['lr_up'].permute(1, 0, 2, 3).contiguous()
            samples['img_rgb'] = samples['img_rgb'].permute(1, 0, 2, 3).contiguous()

        samples = to_device(samples, device)
        norm_img = samples['img_lr'] if args.dataset.lower() in ['fastmri', 'm4raw'] else samples['img_gt']
        samples['img_lr'], samples['lr_up'] = normalize(
            samples['img_lr'], samples['lr_up'], gt_img=norm_img, attr=_test_name, args=args)

        if args.with_noisy:
            noise = torch.FloatTensor(samples['img_lr'].size()).normal_(mean=0, std=5 / 255.).to(device)
            samples['img_lr'] = torch.clamp(samples['img_lr'] + noise, 0, 1)
            samples['lr_up'] = interpolate(samples['img_lr'], scale_factor=args.scale, mode='bicubic', align_corners=False)

        start.record()
        if args.train_supernet and arch is None:
            key = get_random_architecture(args.num_stages, args.num_blocks, args.num_base_ops, args.num_fuse_ops)
            arch = key2arch(key, args.num_stages, args.num_blocks)
            # print(get_model_efficiency(arch, lookup_table, args))
            outputs = model(samples, arch)
        elif arch is not None:
            arch = key2arch(arch, args.num_stages, args.num_blocks) if isinstance(arch, tuple) else arch
            outputs = model(samples, arch)
        else:
            outputs = self_ensemble(samples, model, args.ensemble_mode) if args.self_ensemble else model(samples)
        end.record()
        torch.cuda.synchronize()
        img_out = de_normalize(outputs['img_out'], gt_img=norm_img, attr=_test_name, args=args)


        metric_logger.meters["Time/{}".format(test_name)].update(start.elapsed_time(end), n=nb)
        if args.save_result:
            img_name = samples['img_name']
            for i in range(samples['img_gt'].size(0)):
                np.save('{}/{}.npy'.format(sv_path, img_name[i]), img_out[i].squeeze().detach().cpu().numpy())
    

    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()

    return {k: round(meter.global_avg, 8) for k, meter in metric_logger.meters.items()}