# -*- coding: utf-8 -*-

import time
import torch
from flop_counter.flop_count import FlopCountAnalysis

def calculate_latency(model, inputs):
    start_time = time.time()
    model(inputs)
    latency = time.time() - start_time
    return latency

def calculate_flops(model, inputs):
    flops = FlopCountAnalysis(model, inputs)
    return flops.total()

def calculate_paras(model):
    total_num = sum(p.numel() for p in model.parameters())
    return total_num
