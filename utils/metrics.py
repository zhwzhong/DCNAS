# -*- coding: utf-8 -*-

import torch
import numpy as np
from dist import master_only
from .misc import time_since
import torch.nn.functional as f
from logger import get_root_logger
from loss.pytorch_ssim import ssim
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def mask_rmse(im_pred, im_true, mask, attr):

    std = {'nyu': 1386.05, 'middlebury': 1122.7, 'diml': 1154.29}

    mse = 0.01 * std[attr.lower()]**2 * f.mse_loss(im_pred[mask == 1.], im_true[mask == 1.]).item()
    mae = 0.1 * std[attr.lower()] * f.l1_loss(im_pred[mask == 1.], im_true[mask == 1.]).item()
    return {'RMSE': mse, 'MAE': mae}


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


@torch.no_grad()
def torch_psnr(img1, img2, border=0, data_range=255, qt=False, slice_ssim=False):
    if border != 0:
        img1 = img1[:, :, border: -border, border: -border]
        img2 = img2[:, :, border: -border, border: -border]
    if qt:
        img1, img2 = quantize(img1, data_range), quantize(img2, data_range)

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        _img1 = img1.detach().cpu().numpy().squeeze()
        _img2 = img2.detach().cpu().numpy().squeeze()

        ssim_metric = []
        if slice_ssim:
            for index in range(_img1.shape[0]):

                ssim_metric.append(structural_similarity(_img2[index], _img1[index], data_range=_img2.max()))

            ssim_metric = np.mean(ssim_metric)
        else:
            ssim_metric = structural_similarity(_img2, _img1,  data_range=data_range, channel_axis=0)
        return {'RMSE': 20 * torch.log10_(data_range / torch.sqrt(mse)).item(), 'MAE': ssim_metric}


def torch_rmse(im_pred, im_true, mask, attr):
    border = 0
    attr = attr.lower()
    if attr in ['diml', 'rd']:
        mul_ratio = 0.1
    elif attr in ['middlebury', 'lu', 'sintel', 'mpi', 'a', 'b', 'c', 'sun', 'dut']:
        mul_ratio = 1
    elif attr == 'didoe':
        mul_ratio = 10
    elif attr == 'nyu':
        border = 6
        mul_ratio = 100
    else:
        raise NotImplementedError

    b, c, h, w = im_true.size()
    if border != 0:
        mask = mask.view(b, c, h, w)[:, :, border: -border, border: -border]
        im_pred = im_pred.view(b, c, h, w)[:, :, border: -border, border: -border]
        im_true = im_true.view(b, c, h, w)[:, :, border: -border, border: -border]

    mae = torch.mean(torch.abs((im_true.float()[mask == 1.] - im_pred.float()[mask == 1.]))).item()
    rmse = torch.sqrt(torch.mean((im_true.float()[mask == 1.] - im_pred.float()[mask == 1.]) ** 2)).item() * mul_ratio
    return {'RMSE': rmse, 'MAE': mae}


def metrics(im_pred, im_true, mask, gdata, attr, dataset):
    sum_mae = 0
    sum_rmse = 0
    border = 6 if dataset in ['UAV'] else 0
    for index in range(im_pred.size(0)):
        if gdata:
            metric = mask_rmse(im_pred[index: index + 1], im_true[index: index + 1], mask[index: index + 1], attr)
        elif dataset in ['WV2', 'WV3', 'GF2', 'NIR', 'UAV', 'NIR']:
            metric = torch_psnr(
                im_pred[index: index + 1], im_true[index: index + 1], border=border, data_range=255, qt=True
            )
        elif dataset in ['FastMRI', 'M4Raw']:
            # print('Here', im_pred.size())
            metric = torch_psnr(
                im_pred[index: index + 1], im_true[index: index + 1], border=border,
                data_range=im_true[index: index + 1].max(), slice_ssim=True  # MIN 分开计算SSIM，合到一起算PSNR
            )
        else:
            metric = torch_rmse(im_pred[index: index + 1], im_true[index: index + 1], mask[index: index + 1], attr)
        sum_mae += metric['MAE']
        sum_rmse += metric['RMSE']
    return {'RMSE': sum_rmse / im_pred.size(0), 'MAE': sum_mae / im_pred.size(0)}


def get_mean_max(img_name):
    means = {
        'nyu': 3, 'diml': 2723, 'mpi': 75, 'rd': 1977, 'sintel': 67, 'dut': 37.87156695872226,
        'nir': 111.76465047034439,
        'uav': np.array([58.9369146,  47.9031459,  42.99039452]).reshape(1, 3, 1, 1),
        'a': 124, 'b': 124, 'c':124, 'didoe': 4, 'middlebury': 116, 'lu': 148, 'sun': 90,
        'wv2': np.array([42.6302067, 64.07775558, 43.55840663, 72.92202052]).reshape(1, 4, 1, 1),
        'wv3': np.array([48.49153499, 67.8277034,  51.66618939, 74.27447972]).reshape(1, 4, 1, 1),
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



