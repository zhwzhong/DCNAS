# -*- coding: utf-8 -*-

import os
import time
import math
import json
import torch
import shutil
import random
import itertools
import numpy as np
from dist import master_only

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params based on ordering
        return [p for p in model.parameters()][:-2]
    else:
        return model.parameters()


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_num, trainable_num = clever_format([total_num, trainable_num])
    return {'Total': total_num, 'Trainable': trainable_num}


def to_device(sample, device):
    for key, value in sample.items():
        if key != 'img_name':
            sample[key] = value.to(device, non_blocking=True)
    return sample


def mix_up(samples, alpha, prob=0.7):
    gt_img = samples['img_gt']

    if np.random.rand(1) < prob and alpha > 0:
        lam = np.random.beta(alpha, alpha)
        batch_size = gt_img.size(0)
        index = torch.randperm(batch_size).to(gt_img.device)

        for key, value in samples.items():
            if key != 'img_name':
                samples[key] = lam * value + (1 - lam) * value[index]

    return samples


def transform(*args, xflip, yflip, transpose, reverse=False):
    def _transform(img):
        if not reverse:  # forward transform
            if xflip: img = torch.flip(img, [3])
            if yflip: img = torch.flip(img, [2])
            if transpose: img = torch.transpose(img, 2, 3)
        else:  # reverse transform
            if transpose: img = torch.transpose(img, 2, 3)
            if yflip: img = torch.flip(img, [2])
            if xflip: img = torch.flip(img, [3])
        return img
    out = [_transform(a) for a in args]
    return out if len(out) > 1 else out[0]


def self_ensemble(samples, model, ensemble_mode='mean'):
    outputs = []
    tmp_lr_up = samples['lr_up'].clone()
    tmp_input = samples['img_lr'].clone()
    tmp_color = samples['img_rgb'].clone()
    opts = itertools.product((False, True), (False, True), (False, True))
    for x_flip, y_flip, transpose in opts:
        samples['img_lr'], samples['lr_up'], samples['img_rgb'] = transform(tmp_input.clone(), tmp_lr_up.clone(),
                                                                            tmp_color.clone(), xflip=x_flip, yflip=y_flip,
                                                                            transpose=transpose)
        out_img = model(samples)['img_out']
        outputs.append(transform(out_img, xflip=x_flip, yflip=y_flip, transpose=transpose, reverse=True))

    if ensemble_mode == 'mean':
        out_img = torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        out_img = torch.stack(outputs, 0).median(0)[0]
    else:
        raise ValueError("Unknown ensemble mode %s." % ensemble_mode)
    return {'img_out': out_img}

def pad_height_width(img, scale):
    b, c, h, w = img.size()
    pad_h = scale - h % scale
    pad_w = scale - w % scale
    return pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2

@master_only
def backup_source_code(backup_directory='/model/GDSR'):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)