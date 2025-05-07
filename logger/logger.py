# -*- coding: utf-8 -*-

import os
import shutil
import logging
import logging.handlers
from dist import master_only, get_rank

initialized_logger = {}


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger

def init_tb_loggers(args):
    if args.test_only:
        return None

    if args.log_wandb:
       init_wandb_logger(args)

    tb_logger = init_tb_logger(log_dir=f'{args.log_path}logs/{args.dataset}/{args.file_name}')

    return tb_logger

@master_only
def init_wandb_logger(args):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = get_root_logger()

    resume_id = args.wandb_id
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(id=wandb_id, resume=resume, name=args.file_name, config=args,
               project=args.wandb_project, sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={args.wandb_project}.')

def get_root_logger(logger_name='GDSR', log_level=logging.INFO, write_mode=True, log_file=None):

    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank = get_rank()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w' if write_mode else 'a')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    else:
        logger.setLevel(log_level)
    initialized_logger[logger_name] = True
    return logger

@master_only
def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

@master_only
def set_checkpoint_dir(args, args_text):
    write_mode = True
    log_name = 'GDSR'
    if args.test_only or args.resume or args.generate_train_pair:
        write_mode = False
        log_file = None
        log_name = 'test'
    else:
        create_dir(f'{args.log_path}logs/{args.dataset}/{args.file_name}')
        create_dir(f'{args.log_path}checkpoints/{args.dataset}/{args.file_name}')
        log_file = f'{args.log_path}logs/{args.dataset}/{args.file_name}.log'
        with open(os.path.join(f'{args.log_path}logs/{args.dataset}/{args.file_name}_args.yaml'), 'w') as f:
            f.write(args_text)
    logger = get_root_logger(logger_name=log_name, log_file=log_file, write_mode=write_mode)
    if write_mode:
        logger.info('Removing Previous Checkpoints and Get New Checkpoints Dir')