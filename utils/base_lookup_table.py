# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   base_lookup_table.py
@Time    :   2022/10/8 09:02
@Desc    :
"""
import os
import tqdm
import torch
import logging
import collections
import numpy as np
import pandas as pd
from models import mobile
from models import fusion
from models.common import ConvBNReLU2D
from models.fix_dcnas import StaticModel
from utils import get_random_architecture
from flop_counter.flop_count import FlopCountAnalysis
from utils.architectures import key2arch, get_fix_architecture, arch2key
from models.dcnas import DownSample, UpSample, FeatureInitialization, Tail


class LookUpTable(object):
    def __init__(self, args, file_path):

        self.lookup_table = pd.read_csv(file_path, header=0, index_col=0)
        self.model_statics = pd.read_csv(file_path.replace('.csv', '_static.csv'), header=0, index_col=0)

    def get_max_flops(self):
        return self.model_statics['FLOPs'].max()

    def get_min_flops(self):
        return self.model_statics['FLOPs'].min()

    def get_max_paras(self):
        return self.model_statics['Paras'].max()

    def get_min_paras(self):
        return self.model_statics['Paras'].min()

    def get_max_latency(self):
        return self.model_statics['Latency'].max()

    def get_min_latency(self):
        return self.model_statics['Latency'].min()















