# -*- coding: utf-8 -*-

import glob
import os
import cv2
import h5py
import torch
import numpy as np
from PIL import Image
from data import augment
from torch.utils.data import Dataset
from utils.image_resize import imresize

root_path = None
for path in ['/data/zhwzhong/Data/yaogan', '/data_c/wcy/Data/yaogan', '/home/wcy/Data/yaogan', '/root/autodl-tmp/Data/yaogan']:
    if os.path.exists(path):
        root_path = path
        break


class GF2(Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr =  attr

        self.attr = 'train' if attr == 'train' else 'test'

        if attr.upper() in ['WV2', 'WV3', 'GF2']:
            self.ms_list = sorted(glob.glob(f'{root_path}/{attr.upper()}_data/{self.attr}128/ms/*.tif'))
        else:
            self.ms_list = sorted(glob.glob(f'{root_path}/GF2_data/{self.attr}128/ms/*.tif'))

    def __len__(self):
        return int(self.args.show_every * len(self.ms_list)) if self.attr == 'train' else len(self.ms_list)

    def __getitem__(self, item):
        item = item % len(self.ms_list)
        ms_img = Image.open(self.ms_list[item])
        pan_img = Image.open(self.ms_list[item].replace('ms', 'pan'))

        lms_img = ms_img.resize((int(ms_img.size[0] / self.args.scale),int(ms_img.size[1] / self.args.scale)), Image.BICUBIC)

        lr_up = imresize(np.array(lms_img).astype(np.float32), scalar_scale=self.args.scale).transpose((2, 0, 1))

        pan_img = np.expand_dims(np.array(pan_img), 0)
        ms_img = np.array(ms_img).transpose((2, 0, 1))
        lms_img = np.array(lms_img).transpose((2, 0, 1))

        if self.attr == 'train':
            lms_img, lr_up, ms_img, pan_img = augment.random_rot(lms_img, lr_up, ms_img, pan_img, hflip=True, rot=True)

        lms_img, lr_up, ms_img, pan_img = augment.np_to_tensor(lms_img, lr_up, ms_img, pan_img)

        pan_img = pan_img / 255
        if self.args.img_norm:
            pan_img = (pan_img - 0.1897) / 0.02588

        return {
            'img_gt': ms_img, 'img_rgb': pan_img, 'img_name': os.path.basename(self.ms_list[item]).replace('.tif', ''),
            'img_lr': lms_img, 'lr_up': lr_up, 'img_mask': (~torch.isnan(ms_img)).float(), 'lr_mask': (~torch.isnan(lms_img)).float()
        }


# from config import args
# from utils.metrics import torch_psnr
# from torch.utils.data import DataLoader
# nyu_data = DataLoader(GF2(args, 'test'))
# # nyu_data = DataLoader(NYU(args, 'diml'))
# sum_psnr = []
# for _, sample in enumerate(nyu_data):
#     print(sample['img_gt'].max())
#     sum_psnr.append(torch_psnr(sample['img_gt'], sample['lr_up'])['RMSE'])
#
# print(np.mean(sum_psnr))