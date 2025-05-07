# -*- coding: utf-8 -*-

import os
import torch
import functools
import torch.distributed as dist

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        args.gpu = args.rank % torch.cuda.device_count()
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # print('Not using distributed mode')
        # logger.info('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    if 'WORLD_SIZE' in os.environ:
        args.num_gpus = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    # logger.info('| distributed init (rank {}): {}'.format(args.rank, args.dist_url))
    # print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = get_rank()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
