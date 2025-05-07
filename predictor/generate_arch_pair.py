# -*- coding: utf-8 -*-

import tqdm
import pandas as pd
from predictor.arch_sampler import get_arch_score
from utils import get_random_architecture, key2arch


def generate_train_data(args, num_train_pair, model, eval_data, _logger):
    _logger.info('Generating train archs...')

    visited_arch = {}

    nb = len(range(num_train_pair))
    num_iters = range(num_train_pair)
    if args.local_rank in [-1, 0]:
        num_iters = tqdm.tqdm(num_iters, total=nb)  # 只在主进程打印进度条
    for _ in num_iters:
        arch_cand = get_random_architecture(args.num_stages, args.num_blocks, args.num_base_ops, args.num_fuse_ops)

        while arch_cand in visited_arch.keys():
            _logger.info("Architecture Repeated....")
            arch_cand = get_random_architecture(args.num_stages, args.num_blocks, args.num_base_ops, args.num_fuse_ops)

        arch = key2arch(arch_cand, args.num_stages, args.num_blocks)
        arch_cand = "_".join([str(c) for c in arch_cand])
        visited_arch[arch_cand] = get_arch_score(args, model, 'cuda', arch, eval_data, end_epochs=200)


    visited_arch = pd.DataFrame(visited_arch, index=[0]).T
    visited_arch.to_csv(args.train_arch_path)
    print(visited_arch)