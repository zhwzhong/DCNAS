# -*- coding: utf-8 -*-

import os
import h5py
import torch
import numpy as np
from PIL import Image
from data import augment
from torch.utils.data import Dataset
from data.data_utils import mod_crop, get_lowers


root_path = None



class RD(Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        if attr in ['train', 'val']:
            self.file = h5py.File('{}/{}.h5'.format(root_path, 'rd'))[attr]
        else:
            self.file = h5py.File('{}/test.h5'.format(root_path))['rd']

        func = lambda x: np.array(x) if self.args.cached and attr == 'train' else x
        # func = lambda x: np.array(x).astype(np.float32)
        self.lr_imgs = None
        self.images = [func(self.file['images'].get(key)) for key in self.file['images'].keys()]
        self.depths = [func(self.file['depths'].get(key)) for key in self.file['depths'].keys()]

        self.lr_imgs = [func(self.file['lr'].get(key)) for key in self.file['images'].keys()]

    def __len__(self):
        return int(self.args.show_every * len(self.images)) if self.attr == 'train' else len(self.images)

    def __getitem__(self, item):

        item = item % len(self.images)

        lr = np.array(self.lr_imgs[item]).astype(np.float32)
        dep, rgb = np.array(self.depths[item]).astype(np.float32), np.array(self.images[item]) / 255

        h, w = dep.shape
        lr = np.array(Image.fromarray(lr).resize((w // self.args.scale, h // self.args.scale), Image.BICUBIC))

        lr, dep, rgb = mod_crop(np.expand_dims(lr, 0), np.expand_dims(dep, 0), rgb, modulo=self.args.scale)

        if self.args.img_norm:
            i_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            i_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            rgb = (rgb - i_mean) / i_std

        if self.attr == 'train':
            lr, dep, rgb = augment.get_patch(lr, dep, rgb, patch_size=self.args.patch_size // self.args.scale,
                                             scale=self.args.scale)
            lr, dep, rgb = augment.random_rot(lr, dep, rgb, hflip=True, rot=True)

        lr_up = get_lowers(lr, 1 / self.args.scale, 'bicubic')
        lr, lr_up, dep, rgb = augment.np_to_tensor(lr, lr_up, dep, rgb)

        return {
            'img_gt': dep, 'img_rgb': rgb, 'img_name': str(item),
            'img_lr': lr, 'lr_up': lr_up, 'img_mask': (~torch.isnan(dep)).float(), 'lr_mask': (~torch.isnan(lr)).float()
        }

