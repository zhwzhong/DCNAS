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


# import imageio
# from config import args
# from utils import metrics
# from torch.utils.data import DataLoader
#
# import numpy as np
#
#
# def calc_rmse(out_img, gt_img, border=0, multp=1):
#     gt_img = gt_img.astype(np.float64)
#     out_img = out_img.astype(np.float64)
#     diff = gt_img * multp - out_img * multp
#     if border != 0:
#         diff = diff[border: -border, border: -border]
#     return round(np.sqrt(np.mean(np.power(diff, 2))), 2)
#
# nyu_data = DataLoader(NYU(args, 'nyu'))
# sum_rmse = []
# sum_rmse_Np = []
# # nyu_data = DataLoader(NYU(args, 'diml'))
# for _, sample in enumerate(nyu_data):
#
#     rmse = metrics(
#         sample['lr_up'], sample['img_gt'], sample['img_mask'], False, attr='nyu', dataset=args.dataset)['RMSE']
#     sum_rmse.append(rmse)
#     sum_rmse_Np.append(calc_rmse(sample['lr_up'].cpu().numpy().squeeze(), sample['img_gt'].cpu().numpy().squeeze(), border=6, multp=100))
# print(np.mean(np.array(sum_rmse)))
# print(np.mean(np.array(sum_rmse_Np)))


