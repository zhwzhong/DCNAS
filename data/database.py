# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
from torch.utils.data import Dataset

root_path = None
for path in ['/data/zhwzhong/Data/GDSR', '/data_c/wcy/Data/GDSR', '/home/wcy/Data/GDSR', '/root/autodl-tmp/Data/GDSR',
             '/home/zhizhong/Data/GDSR']:
    tmp_path = '{}/data'.format(path)
    if os.path.exists(tmp_path):
        root_path = tmp_path
        break


class Base(Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        if attr in ['train', 'val']:
            self.file = h5py.File('{}/{}.h5'.format(root_path, args.dataset.lower()))[attr]
        else:
            if self.args.dataset.lower() == 'rd':
                self.attr = 'RD'
            self.file = h5py.File('{}/test.h5'.format(root_path))[self.attr.lower()]

        func = lambda x: np.array(x) if self.args.cached and attr == 'train' else x
        # func = lambda x: np.array(x).astype(np.float32)
        self.lr_imgs = None
        self.images = [func(self.file['images'].get(key)) for key in self.file['images'].keys()]
        self.depths = [func(self.file['depths'].get(key)) for key in self.file['depths'].keys()]

        if self.args.dataset.lower() == 'rd':
            self.lr_imgs = [func(self.file['lr'].get(key)) for key in self.file['images'].keys()]

    def __len__(self):
        return int(self.args.show_every * len(self.images)) if self.attr == 'train' else len(self.images)

    def __getitem__(self, item):
        pass

