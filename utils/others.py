# -*- coding: utf-8 -*-

import tqdm
import data
import numpy as np
from config import args
from models.guided_filter import GuidedFilter
from utils import normalize, de_normalize, metrics
from torchvision.transforms.functional import rgb_to_grayscale



if __name__ == '__main__':
    # args.dataset = 'RD'
    # args.scale = 2
    # args.patch_size = 32
    # val_data = data.get_loader(args, 'train')
    # for _, sample in enumerate(val_data):
    #     print(sample['lr_up'].shape)
    rd_data()
    # get_bic_rmse()
    # d_data_rmse()