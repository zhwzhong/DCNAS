# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   __init__.py.py
@Time    :   2022/7/15 09:18
@Desc    :
"""
import os
from .nyu import NYU
from .diml import DIML
from .middlebury import Middlebury
from torchvision.transforms import Normalize


def get_dataset(args, attr, data_dir):
    crop_size = args.patch_size if attr == 'train' else 256
    crop_deterministic = False if attr == 'train' else True
    data_args = {
        'crop_size': (crop_size, crop_size),
        'in_memory': True,
        'max_rotation_angle': 15. if attr == 'train' else 0,
        'do_horizontal_flip': True if attr == 'train' else False,
        'crop_valid': True,
        'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        'scaling': args.scale
    }

    if args.dataset == 'Middlebury':
        depth_transform = Normalize([2296.78], [1122.7])
        datasets = Middlebury(os.path.join(data_dir, 'Middlebury'), **data_args, split=attr,
                              depth_transform=depth_transform, crop_deterministic=crop_deterministic)

    elif args.dataset == 'DIML':
        depth_transform = Normalize([2749.64], [1154.29])
        datasets = DIML(os.path.join(data_dir, 'DIML'), **data_args, split=attr,
                        depth_transform=depth_transform, crop_deterministic=crop_deterministic)

    elif args.dataset == 'NYU':
        depth_transform = Normalize([2796.32], [1386.05])
        datasets = NYU(os.path.join(data_dir, 'NYUv2'), **data_args, split=attr,
                       depth_transform=depth_transform, crop_deterministic=crop_deterministic)
    else:
        raise NotImplementedError(f'Dataset {args.dataset}')
    return datasets