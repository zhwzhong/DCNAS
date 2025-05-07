# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   __init__.py.py
@Time    :   2022/7/8 10:53
@Desc    :
"""
import os
import torch
import pickle
from utils.cuda import ApexScaler
from logger import get_root_logger
from importlib import import_module
from utils.architectures import key2arch
from utils.optimizer import make_optimizer
from utils.misc import get_parameter_number
from utils.checkpoiont import resume_checkpoint

try:
    from apex import amp
    from apex.parallel import convert_syncbn_model
    from apex.parallel import DistributedDataParallel as ApexDDP
except:
    pass
from torch.nn.parallel import DistributedDataParallel as NativeDDP


def init_model(model, args):

    device = torch.device(args.device)
    model.to(device)
    optimizer = make_optimizer(args, model)

    _logger = get_root_logger()
    loss_scaler = ApexScaler() if args.apex else None
    # loss_scaler = ApexScaler() if args.apex else NativeScaler()
    if os.path.exists(args.load_name):
        load_path = args.load_name
    elif args.search:
        file_name = args.file_name.replace('Search', 'TrainSuper')
        load_path = f'{args.log_path}checkpoints/{args.dataset}/{file_name}/{args.load_name}'
    else:
        load_path = f'{args.log_path}checkpoints/{args.dataset}/{args.file_name}/{args.load_name}'

    if os.path.exists(load_path):
        args.start_epoch = resume_checkpoint(
            model, load_path,
            optimizer=optimizer if args.resume else None,
            loss_scaler=loss_scaler if args.resume else None,
            log_info=args.local_rank == 0)
        if args.local_rank == 0 and args.start_epoch > 0:
            if args.test_only:
                _logger.info('Scheduled epochs: {}'.format(args.start_epoch - 1))
            else:
                _logger.info('Scheduled epochs: {}'.format(args.start_epoch))

    if args.distributed:
        if args.apex:
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")

            if args.sync_bn:
                model = convert_syncbn_model(model)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            find_unused_parameters = args.model in ['DCNAS', 'FDSR', 'DAGF', 'SVLRM', 'PMBAN', 'DCTNet', 'CODON', 'CODOW', 'PTIR']
            # if args.model == 'DCNAS' and args.train_random:
            #     find_unused_parameters = False
            model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters=find_unused_parameters) # , find_unused_parameters=True
    else:
        model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus))).to(device)
        if args.local_rank == 0:
            _logger.info("Using native Torch DataParallel.")
    if args.local_rank == 0:
        _logger.info("Number of Parameters: {}".format(get_parameter_number(model.module)))
    if args.static_mode:
        model = torch.compile(model, mode="max-autotune")
    return model, optimizer, loss_scaler


def get_model(args):
    module = import_module('models.' + args.model.lower())

    if args.model.lower() == 'dcnas':
        if args.load_from_json:
            with open('{}_{}_{}.pkl'.format(args.dataset.lower(), args.num_features, args.scale), 'rb') as f:
                arch = pickle.load(f)
            print(arch)
            return init_model(module.make_model(args, arch), args)

        elif args.train_random:
            file_name = args.file_name.replace('TrainRandom', 'Search')
            load_path = f'{args.log_path}checkpoints/{args.dataset}/{file_name}/last.pth.tar'
            if os.path.exists(load_path):
                cand = torch.load(load_path)['keep_top_k'][args.select_num][0]
                arch = key2arch(cand, args.num_stages, args.num_blocks)
                print('===> Model Loaded from {}'.format(load_path))
                print(arch)
                return init_model(module.make_model(args, arch), args)
            else:
                print(f'Can\'t load file {load_path}...')
                raise FileNotFoundError
        else:
            return init_model(module.make_model(args, None), args)
    else:
        return init_model(module.make_model(args), args)
