# -*- coding: utf-8 -*-

import os
import cv2
import h5py
import torch
import numpy as np
from data import augment
from torch.utils.data import Dataset
from utils.image_resize import imresize


root_path = None



def get_array(x, cached):
    return np.array(x) if cached else x


class NIR(Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        file_name = f'nir_test_x{args.scale}.h5' if attr == 'test' else 'nir_train.h5'
        self.file = h5py.File(f'{root_path}/{file_name}', 'r')[attr]

        cached = (self.args.cached and attr == 'train')

        self.img_names = [key for key in self.file['GT'].keys()]

        self.gt_imgs = [get_array(self.file['GT'].get(key), cached=cached) for key in self.img_names]
        self.rgb_imgs = [get_array(self.file['RGB'].get(key), cached=cached) for key in self.img_names]

    def __len__(self):
        return int(self.args.show_every * len(self.img_names)) if self.attr == 'train' else len(self.img_names)

    def __getitem__(self, item):
        item = item % len(self.gt_imgs)

        gt_img, rgb_img = np.array(self.gt_imgs[item]), np.array(self.rgb_imgs[item])

        if self.attr == 'test':
            lr_img = gt_img
            gt_img = imresize(lr_img.astype(float), scalar_scale=self.args.scale)
        else:
            if self.args.degration == 'BD':
                scale = np.random.uniform(1., self.args.scale) if self.attr == 'train' else (self.args.scale + 1) / 2

                tmp_gt = cv2.GaussianBlur(gt_img, (15, 15), scale).astype(np.uint8)
            else:
                tmp_gt = gt_img
            lr_img = cv2.resize(tmp_gt, fx=1 / self.args.scale, fy=1 / self.args.scale, dsize=None)

            if self.args.degration == 'BD':
                lr_img += np.random.normal(0, 5, lr_img.shape).astype(np.uint8)

        lr_up = imresize(lr_img.astype(float), scalar_scale=self.args.scale)

        lr_up = np.transpose(lr_up, (2, 0, 1))
        lr_img = np.transpose(lr_img, (2, 0, 1))
        gt_img = np.transpose(gt_img, (2, 0, 1))
        rgb_img = np.transpose(rgb_img, (2, 0, 1)) / 255

        if self.attr == 'train':
            lr_img, lr_up, gt_img, rgb_img = augment.get_patch(
                lr_img, lr_up, gt_img, rgb_img, patch_size=self.args.patch_size, scale=self.args.scale,
            )
            lr_img, lr_up, gt_img, rgb_img = augment.random_rot(
                lr_img, lr_up, gt_img, rgb_img, hflip=True, rot=True
            )

        if self.args.img_norm:
            i_std = np.array([0.22773795, 0.22367531, 0.26343636]).reshape(3, 1, 1)
            i_mean = np.array([0.45617331, 0.49676442, 0.44097971]).reshape(3, 1, 1)
            rgb_img = (rgb_img - i_mean) / i_std

        lr_img, lr_up, gt_img, rgb_img = augment.np_to_tensor(lr_img, lr_up, gt_img, rgb_img, input_data_range=1)
        # print(lr_up.size(), lr_up.size(), gt_img.size(), rgb_img.size())

        lr_img, lr_up, gt_img = lr_img[0: 1], lr_up[0: 1], gt_img[0: 1]
        return {
            'img_gt': gt_img, 'img_rgb': rgb_img, 'lr_up': lr_up, 'img_name': self.img_names[item], 'img_lr': lr_img,
            'img_mask': (~torch.isnan(gt_img)).float(), 'lr_mask': (~torch.isnan(lr_img)).float()
        }

