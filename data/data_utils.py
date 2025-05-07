# -*- coding: utf-8 -*-

import math
import torch
import random
import numpy as np
from PIL import Image
from typing import Sequence
from skimage.transform import resize
import xml.etree.ElementTree as etree


def get_patch(*args, patch_size=32, scale=1):
    """
    :param args: (LR, HR, ..)
    :param patch_size: LR Patch Size
    :param scale: HR // LR
    :return: (LR, HR, ..)
    """
    ih, iw = args[0].shape[1:]
    tp = scale * patch_size
    ip = tp // scale

    iy = random.randrange(0, ih - ip + 1)
    ix = random.randrange(0, iw - ip + 1)
    tx, ty = scale * ix, scale * iy
    ret = [
        args[0][:, iy:iy + ip, ix:ix + ip],
        *[a[:, ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return ret if len(ret) > 1 else ret[0]


def random_rot(*args, hflip=True, rot=True):
    """
    Input: (C, H, W)
    :param args:
    :param hflip:
    :param rot:
    :return:
    """
    hflip = hflip and random.random() < 0.5
    vflip = hflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, :, ::-1]
        if vflip: img = img[:, ::-1, :]
        if rot90: img = img.transpose(0, 2, 1)
        return np.ascontiguousarray(img)

    out = [_augment(a) for a in args]
    return out if len(out) > 1 else out[0]


def np_to_tensor(*args, input_data_range=1.0, process_data_range=1.0):
    def _np_to_tensor(img):
        np_transpose = img.astype(np.float32)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(process_data_range / input_data_range)
        return tensor.float()

    out = [_np_to_tensor(a) for a in args]
    return out if len(out) > 1 else out[0]

def get_lowers(im_np, factor, mode='last'):
    """
    mode: 'bicubic', 'bilinear', 'nearest', 'last', 'center'
    """
    if im_np.ndim == 3:
        im_np = im_np.transpose(1, 2, 0)

    h0, w0 = im_np.shape[:2]
    h, w = int(math.ceil(h0 / float(factor))), int(math.ceil(w0 / float(factor)))

    if h0 != h * factor or w0 != w * factor:
        im_np = resize(im_np, (h * factor, w * factor), order=1, mode='reflect', clip=False, preserve_range=True,
                       anti_aliasing=True)
    if mode == 'nearest':

        idxs = (slice(factor - 1, None, factor),) * 2

        lowers = im_np[idxs].copy()
    else:
        if len(im_np.shape) == 3:
            im_np = im_np[:, :, 0]
            lowers = np.expand_dims(np.array(Image.fromarray(im_np).resize((w, h), Image.BICUBIC)), 2)
        else:
            lowers = np.array(Image.fromarray(im_np).resize((w, h),Image.BICUBIC))
    if lowers.ndim == 3:
        lowers = lowers.transpose((2, 0, 1))

    return lowers


def mod_crop(*args, modulo):
    def _mod_crop(img):
        if len(img.shape) == 2:
            h, w = img.shape
            crop_h, crop_w = h % modulo, w % modulo
            return img[crop_h // 2: h - (crop_h - crop_h // 2), crop_w // 2: w - (crop_w - crop_w // 2)]
        else:
            _, h, w = img.shape
            crop_h, crop_w = h % modulo, w % modulo
            return  img[:, crop_h // 2: h - (crop_h - crop_h // 2), crop_w // 2: w - (crop_w - crop_w // 2)]

    out = [_mod_crop(a) for a in args]
    return out if len(out) > 1 else out[0]


def padding(dep, rgb, h, w):
    # 获取输入图像的尺寸（C, H, W）
    _, dep_h, dep_w = dep.shape
    _, rgb_h, rgb_w = rgb.shape

    # 如果深度图像的高度或宽度小于目标尺寸，则计算需要填充的像素
    if dep_h < h or dep_w < w:
        top = (h - dep_h) // 2 if dep_h < h else 0
        bottom = h - dep_h - top if dep_h < h else 0
        left = (w - dep_w) // 2 if dep_w < w else 0
        right = w - dep_w - left if dep_w < w else 0
        dep = np.pad(dep, ((0, 0), (top, bottom), (left, right)), mode='edge')

    # 如果 RGB 图像的高度或宽度小于目标尺寸，则计算需要填充的像素
    if rgb_h < h or rgb_w < w:
        top = (h - rgb_h) // 2 if rgb_h < h else 0
        bottom = h - rgb_h - top if rgb_h < h else 0
        left = (w - rgb_w) // 2 if rgb_w < w else 0
        right = w - rgb_w - left if rgb_w < w else 0
        rgb = np.pad(rgb, ((0, 0), (top, bottom), (left, right)), mode='edge')

    return dep, rgb

