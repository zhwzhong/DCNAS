# -*- coding: utf-8 -*-

from config import args  # 必须在第一个

import os
import torch
from time import time
from loss import Loss
from data import get_loader
from models import get_model
from predictor import Estimator
from search import EvolutionSearcher
from scheduler import create_scheduler
from predictor.pdata import ArchData
from torch.utils.data import DataLoader
from predictor import generate_train_data
from flop_counter import FlopCountAnalysis
from trainer import train_one_epoch, evaluate
from logger import get_root_logger, init_tb_loggers


from utils import set_random_seed, CheckpointSaver, update_summary, key2arch, resume_checkpoint
from utils.base_lookup_table import get_model_efficiency, calculate_model_efficiency, LookUpTable


def main():
    set_random_seed(args.seed)
    _logger = get_root_logger()
    tb_logger = init_tb_loggers(args)

    criterion = Loss(args)
    model, optimizer, loss_scaler = get_model(args)
    # print(model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    lr_scheduler.step(args.start_epoch)
    cp_path = f'{args.log_path}checkpoints/{args.dataset}/{args.file_name}'

    decreasing = False if args.dataset in ['WV2', 'GF2', 'NIR'] else True
    saver = CheckpointSaver(
        model=model, optimizer=optimizer, args=args, amp_scaler=loss_scaler,
        checkpoint_dir=cp_path, recovery_dir=cp_path, decreasing=decreasing, max_history=args.checkpoint_hist
    )

    lookup_table = None
    eval_data = get_loader(args, 'val')

    estimator = None
    if args.model == 'DCNAS':
        estimator = Estimator(
            num_base_ops=args.num_base_ops, num_fuse_ops=args.num_fuse_ops,
            embed_size=32, hidden_size=32, num_layers=2, lr=1e-4, device=args.device,
        )
        if os.path.exists(args.model_path):
            if args.local_rank == 0:
                estimator.load(args.model_path)
                _logger.info('Performance Estimator Loaded ...')
        lookup_table = LookUpTable(args, args.lookup_table_path)

    if args.generate_train_pair:
        return generate_train_data(args, args.num_train_pair, model, eval_data, _logger)

    if args.resume or args.test_only or args.pre_train:
        for test_name in args.test_set:
            val_data = get_loader(args, test_name)
            test_stats = evaluate(model, val_data, test_name, args=args)
            _logger.info(test_stats)

    # if args.train_random:
    #     file_name = args.file_name.replace('TrainRandom', 'Search')
    #     load_path = f'{args.log_path}checkpoints/{args.dataset}/{file_name}/last.pth.tar'
    #     if os.path.exists(load_path) and args.local_rank == 0:
    #         checkpoint = torch.load(load_path)
    #         cand = checkpoint['keep_top_k'][args.select_num][0]
    #         arch = key2arch(cand, args.num_stages, args.num_blocks)
    #         comp = calculate_model_efficiency(
    #             model, 1, args.in_channels, args.guidance_channels, 480, 480, device=torch.device(args.device)
    #         )
    #
    #         _logger.info('Flops: {}, Paras: {}, Latency: {}'.format(comp['FLOPs'], comp['Paras'], comp['Latency']))
    #         comp = get_model_efficiency(arch, lookup_table.lookup_table, args)
    #         _logger.info('My Flops: {}, Paras: {}, Latency: {}'.format(comp['FLOPs'], comp['Paras'], comp['Latency']))
    if args.search:
        searcher = EvolutionSearcher(args, model, evaluate, _logger, tb_logger, lookup_table.lookup_table)
        searcher.search()
    elif not args.test_only:
        best_epoch = None
        best_metric = None
        start_time = time()
        train_data = get_loader(args, 'train')
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_data.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, train_data, eval_data, lookup_table, optimizer, epoch, args, loss_scaler, estimator
            )

            if ((epoch % args.finetune_interval == 0) and epoch > 30 and args.model == 'DCNAS'
                    and args.train_supernet and args.sampling == 'P'):
                generate_train_data(args, 100, model, train_data, _logger)
                train_loader = DataLoader(ArchData(file_path=args.train_arch_path), batch_size=16, shuffle=True)
                estimator.fine_tune(train_loader, num_epochs=50)
                estimator.save(path=args.model_path)

            log_stats = {**{f'Train/{k}'.upper(): v for k, v in train_stats.items()}}
            lr_scheduler.step(epoch)
            test_name = "VAL"
            for test_name in args.test_set:
                val_data = get_loader(args, test_name)
                test_stats = evaluate(model, val_data, test_name, args=args, logger=_logger)
                log_stats.update({**{f'{k}'.upper(): v for k, v in test_stats.items()}})
            update_summary(epoch, start_time, log_stats, tb_logger)

            if args.local_rank == 0:
                save_metric = log_stats[f'RMSE/{test_name}'.upper()]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
        if best_metric is not None:
            _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    else:
        return False


if __name__ == '__main__':
    main()