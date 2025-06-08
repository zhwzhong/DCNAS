# -*- coding: utf-8 -*-

import os
import yaml
import argparse
from config import set_config
from logger import set_checkpoint_dir
from dist import init_distributed_mode
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch GDSR Training')

# Hardware
parser.add_argument('--device', default='cuda')
parser.add_argument('--apex', action='store_true')
parser.add_argument('--sync_bn', action='store_true')
parser.add_argument('--static_mode', action='store_true')
parser.add_argument('--seed', type=int, default=60)
parser.add_argument('--num_gpus', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)

# Dataset
parser.add_argument('--gdata', action='store_true')
parser.add_argument('--dataset', type=str, default='NYU')
parser.add_argument('--show_every', type=float, default=1)
parser.add_argument('--print_freq', type=int, default=1000)
parser.add_argument('--repeated_aug', action='store_true')
# parser.add_argument('--data_augment', action='store_true')
parser.add_argument('--normalize',  type=int, default=0)
parser.add_argument('--degration', type=str, default='BI')
parser.add_argument('--down_type', type=str, default='bicubic')

parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--guidance_channels', type=int, default=3)

parser.add_argument('--cached', action='store_true')


# Training Stats

parser.add_argument('--resume', action='store_true')
parser.add_argument('--pre_train', action='store_true')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--log_path', type=str, default='./')
parser.add_argument('--checkpoint_hist', type=int, default=200)
parser.add_argument('--load_name', type=str, default='model_best.pth.tar')

# Optimizer

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--opt', type=str, default='Adam')
parser.add_argument('--loss', type=str, default='1*L1')
parser.add_argument('--hdelta', type=float, default=1)


parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--sched', default='step', type=str)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--cooldown_epochs', type=int, default=10)

parser.add_argument('--min_lr', type=float, default=1e-10)
parser.add_argument('--warmup_lr', type=float, default=1e-8)
parser.add_argument('--decay_rate',  type=float, default=0.5)
parser.add_argument('--decay_epochs', type=str, default='100')

# Train Config

parser.add_argument('--with_noisy', action='store_true')

parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--patch_size', type=int, default=224)
parser.add_argument('--val_ratio', type=float, default=0.1)
parser.add_argument('--val_batch_size', type=int, default=4)


parser.add_argument('--mix_up', action='store_true')
parser.add_argument('--alpha', type=float, default=10)

# Model config
parser.add_argument('--num_dec_layers', type=int, default=3)
parser.add_argument('--num_color_layers', type=int, default=3)
parser.add_argument('--num_depth_layers', type=int, default=3)
parser.add_argument('--fuse_layer', type=str, default='1_2_3')

# DCNAS Config
parser.add_argument('--norm', default=None)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--act', type=str, default='PReLU')
parser.add_argument('--num_blocks', type=int, default=4)
parser.add_argument('--num_stages', type=int, default=4)
parser.add_argument('--num_features', type=int, default=16)

# DCNAS
parser.add_argument('--model_path', type=str, default='') 
parser.add_argument('--train_arch_path', type=str, default='')
parser.add_argument('--lookup_table_path', type=str, default='')
parser.add_argument('--estimator_lr', type=float, default=1e-4)

# Rank Loss
parser.add_argument('--rank_sign', action='store_true')
parser.add_argument('--rank_sigma', type=float, default=1)
parser.add_argument('--num_archs', type=int, default=20)
parser.add_argument('--sampling', type=str, default='U')
parser.add_argument('--sample_interval', type=int, default=1)
parser.add_argument('--finetune_interval', type=int, default=20)
parser.add_argument('--generate_train_pair', action='store_true')
parser.add_argument('--num_train_pair', type=int, default=1024)

# For DAGF
parser.add_argument('--py_loss', action='store_true')
parser.add_argument('--num_res', type=int, default=3)
parser.add_argument('--num_pyramid', type=int, default=3)
parser.add_argument('--filter_size', type=int, default=3)

parser.add_argument('--num_fuse_ops', type=int, default=7)
parser.add_argument('--num_base_ops', type=int, default=16)

parser.add_argument('--search', action='store_true')
parser.add_argument('--init_arch', action='store_true')
parser.add_argument('--random_search', action='store_true')

parser.add_argument('--train_random', action='store_true')
parser.add_argument('--train_supernet', action='store_true')
parser.add_argument('--load_from_json', action='store_true')
parser.add_argument('--generate_lookup_table', action='store_true')

parser.add_argument('--fix_rgb', type=str, default='')
parser.add_argument('--fix_dep', type=str, default='')
parser.add_argument('--fix_dec', type=str, default='')
parser.add_argument('--fix_fusion', type=str, default='')


parser.add_argument('--flop_max', type=float, default=-1)   # x1e10
parser.add_argument('--para_max', type=float, default=1.2)   # 1e6
parser.add_argument('--latency_max', type=float, default=-1)   # 10

parser.add_argument('--m_prob', type=float, default=0.2)
parser.add_argument('--select_num', type=int, default=10)
parser.add_argument('--mutation_num', type=int, default=25)
parser.add_argument('--population_num', type=int, default=50)
parser.add_argument('--crossover_num', type=int, default=25)

# Augment 的参数
parser.add_argument('--img_norm', action='store_false')
parser.add_argument('--color_augment', action='store_true')

parser.add_argument('--self_ensemble', action='store_true')
parser.add_argument('--ensemble_mode', type=str, default='mean')

parser.add_argument('--log_wandb', action='store_true')
parser.add_argument('--wandb_id', type=str, default='')
parser.add_argument('--wandb_project', type=str, default='GDSR')


parser.add_argument('--model', type=str, default='DCNAS')


parser.add_argument('--clip-grad', type=float, default=None)
parser.add_argument('--clip-mode', type=str, default='norm')


parser.add_argument('--dist_url', default='env://')
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--save_result', action='store_true')
parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--test_set', type=str, default='val')

args_config, remaining = config_parser.parse_known_args()

if args_config.config:
    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

args = parser.parse_args(remaining)

set_config(args)
args.test_set = args.test_set.split('_')

if args.sched == 'multistep':
    args.decay_epochs = [int(x) for x in args.decay_epochs.split('_')]
else:
    args.decay_epochs = int(args.decay_epochs)

args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

init_distributed_mode(args)
set_checkpoint_dir(args, args_text)

if args.local_rank == 0:
    print(args)