# -*- coding: utf-8 -*-

import torch
import numpy as np
from data import augment
from data.database import Base
from data.data_utils import get_lowers, mod_crop, padding


class NYU(Base):

    def __getitem__(self, item):

        item = item % len(self.images)

        dep, rgb = np.array(self.depths[item]).astype(np.float32), np.array(self.images[item])

        dep, rgb = mod_crop(np.expand_dims(dep, 0), rgb, modulo=32)

        # if self.args.patch_size == 640:
        #     dep, rgb = padding(dep, rgb, 640, 640)

        if self.attr == 'train':
            dep, rgb = augment.get_patch(dep, rgb, patch_size=self.args.patch_size)
            dep, rgb = augment.random_rot(dep, rgb, hflip=True, rot=True)

            if self.args.color_augment:
                rgb = augment.color_augment(rgb)

        lr = get_lowers(dep, self.args.scale, self.args.down_type)

        lr_up = get_lowers(lr, 1 / self.args.scale, 'bicubic')

        rgb = rgb / 255
        if self.args.img_norm:
            i_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            i_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            rgb = (rgb - i_mean) / i_std

        lr, lr_up, dep, rgb = augment.np_to_tensor(lr, lr_up, dep, rgb)

        return {
            'img_gt': dep, 'img_rgb': rgb, 'img_name': str(item),
            'img_lr': lr, 'lr_up': lr_up, 'img_mask': (~torch.isnan(dep)).float(), 'lr_mask': (~torch.isnan(lr)).float()
        }


