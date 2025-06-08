# -*- coding: utf-8 -*-

import torch
import numpy as np
from dist import master_only
from .misc import time_since
import torch.nn.functional as f
from logger import get_root_logger
from loss.pytorch_ssim import ssim
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def get_mean_max(img_name):
    means = {
        'nyu': 3, 'diml': 2723, 'mpi': 75, 'rd': 1977, 'sintel': 67, 'dut': 37.87156695872226,
        'nir': 111.76465047034439,
        'uav': np.array([58.9369146,  47.9031459,  42.99039452]).reshape(1, 3, 1, 1),
        'a': 124, 'b': 124, 'c':124, 'didoe': 4, 'middlebury': 116, 'lu': 148, 'sun': 90,
        'wv2': np.array([42.6302067, 64.07775558, 43.55840663, 72.92202052]).reshape(1, 4, 1, 1),
        'gf2': np.array([104.73031355, 71.25249889, 32.50238111, 54.13954554]).reshape(1, 4, 1, 1)
    }
    maxs = {
        'nyu': 9.99547, 'diml': 6118, 'mpi': 255, 'rd': 4466, 'sintel': 255, 'wv2': 255, 'gf2': 255, 'wv3': 255,
        'dut': 255, 'a': 255, 'b': 255, 'c': 255, 'didoe': 255, 'middlebury': 232, 'lu': 255, 'sun': 255, 'uav': 255,
        'nir': 255
    }
    return means[img_name.lower()], maxs[img_name.lower()]


@master_only
def update_summary(epoch, start_time, log_stats, tb_logger):
    logger = get_root_logger()
    try:
        log_info = f"Epoch: [{epoch}], Loss {log_stats['TRAIN/LOSS']:.5f}, "
    except KeyError:
        log_info = f"Epoch: [{epoch}], "
    for k, v in log_stats.items():
        tb_logger.add_scalar(k, v, epoch)
        if k.find('RMSE') != -1:
            log_info += f"{k}: {v:.8f}, "
    # print(log_stats)
    # for name, param in model.named_parameters():
    #     tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    log_info += f"Time Spend {time_since(start_time)}"
    logger.info(log_info)


def normalize(*imgs, gt_img, attr, args):
    b, c, h, w = gt_img.size()
    device = 'cpu' if gt_img.get_device() == -1 else gt_img.get_device()

    def _normalize(img):

        if args.normalize == 0:   # (img - min(GT)) / (max(GT) - nim(GT))
            img_min, _ = torch.min(gt_img.view(b, c, -1, 1), dim=2, keepdim=True)
            img_max, _ = torch.max(gt_img.view(b, c, -1, 1), dim=2, keepdim=True)
            img_max = img_max - img_min
        elif args.normalize == 1:  # (img -mean) / max
            img_min, img_max = get_mean_max(attr)

            img_min = torch.tensor(img_min, device=device)
            img_max = torch.tensor(img_max, device=device)
        elif args.normalize == 2:
            img_min = torch.mean(gt_img.view(b, c, -1, 1), dim=2, keepdim=True)
            img_max = torch.std(gt_img.view(b, c, -1, 1), dim=2, keepdim=True)
        else:                     # img / max
            _, img_max = get_mean_max(attr)
            img_min = torch.tensor(0, device=device)
            img_max = torch.tensor(img_max, device=device)
        # print(img_min.shape, img_max.shape, img.shape, ((img - img_min) / img_max).shape, 'iii')
        norm_img = (img - img_min) / img_max
        return norm_img.clamp(-6, 6) if args.dataset.lower() in ['fastmri', 'm4raw'] else norm_img

    out = [img if args.gdata else _normalize(img) for img in imgs]
    return out if len(out) > 1 else out[0]


def de_normalize(*imgs, gt_img, attr, args):

    b, c, h, w = gt_img.size()
    device = 'cpu' if gt_img.get_device() == -1 else gt_img.get_device()

    def _de_normalize(img):
        if args.normalize == 0:  #  (img - min(GT)) / (max(GT) - nim(GT))
            img_min, _ = torch.min(gt_img.view(b, c, -1, 1), dim=2, keepdim=True)
            img_max, _ = torch.max(gt_img.view(b, c, -1, 1), dim=2, keepdim=True)
            img_max = img_max - img_min
        elif args.normalize == 1: # (img -mean) / max
            img_min, img_max = get_mean_max(attr)
            img_min = torch.tensor(img_min, device=device)
            img_max = torch.tensor(img_max, device=device)
        elif args.normalize == 2:
            img_min = torch.mean(gt_img.view(b, c, -1, 1), dim=2, keepdim=True)
            img_max = torch.std(gt_img.view(b, c, -1, 1), dim=2, keepdim=True)
        else:                     # img / max
            _, img_max = get_mean_max(attr)
            img_min = torch.tensor(0, device=device)
            img_max = torch.tensor(img_max, device=device)

        return img * img_max + img_min

    out = [img if args.gdata else _de_normalize(img) for img in imgs]

    return out if len(out) > 1 else out[0]



