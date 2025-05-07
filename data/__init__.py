# -*- coding: utf-8 -*-

import os
from .nyu import NYU
from gdata import get_dataset
from data.samplers import RASampler
from importlib import import_module
from prefetch_generator import BackgroundGenerator
from dist import get_rank, get_world_size
from torch.utils.data import DistributedSampler, DataLoader


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_loader(args, attr):
    if args.gdata:
        data_dir = '/data/zhwzhong/Data/GDSR/GraphSR/dataset'
        if os.path.exists('/home/zhwzhong/Data/GDSR/GraphSR/dataset'):
            data_dir = '/home/zhwzhong/Data/GDSR/GraphSR/dataset'
        dataset = get_dataset(args, attr, data_dir=data_dir)
    else:
        dataset = getattr(import_module('data.' + args.dataset.lower()), args.dataset)(args, attr)

    global_rank = get_rank()
    num_task = get_world_size()
    if args.distributed and attr == 'train':

        if args.repeated_aug:  # 一个mini-batch中可以包含来自同一个图像的不同增强版本
            data_sampler = RASampler(dataset, num_replicas=num_task, rank=global_rank, shuffle=True)
        else:
            data_sampler = DistributedSampler(dataset, num_replicas=num_task, rank=global_rank, shuffle=True)
    else:
        # data_sampler = None
        data_sampler = DistributedSampler(dataset, num_replicas=num_task, rank=global_rank, shuffle=False)

    shuffle = (data_sampler is None) and (attr == 'train')
    batch_size = args.batch_size if attr == 'train' else args.val_batch_size
    batch_size = max(int(batch_size // args.num_gpus), 1) if args.distributed else batch_size
    batch_size = 1 if attr.lower() == 'middlebury' else batch_size
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=data_sampler, pin_memory=False,
        drop_last=(attr == 'train'), shuffle=shuffle
    )
    return data_loader
