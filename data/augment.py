# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T


def get_patch(*args, patch_size=32, scale=1):
    """
    :param args: (LR, HR, ..)
    :param patch_size: HR Patch Size
    :param scale: HR // LR
    :return: (LR, HR, ..)
    """
    ih, iw = args[0].shape[1:]
    tp = patch_size
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


def color_augment(rgb):
    """

    :param rgb: (C, H, W)
    :return: (C, H, W)
    """
    rgb = rgb.transpose((1, 2, 0))
    rgb = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
    rgb_transform = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    ])
    rgb = np.array(rgb_transform(rgb)).transpose((2, 0, 1))
    return rgb


def np_to_tensor(*args, input_data_range=1.0, process_data_range=1.0):
    def _np_to_tensor(img):
        np_transpose = img.astype(np.float32)
        if len(np_transpose.shape) == 2:
            np_transpose = np.expand_dims(np_transpose, 0)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(process_data_range / input_data_range)
        return tensor.float()

    out = [_np_to_tensor(a) for a in args]
    return out if len(out) > 1 else out[0]