# -*- coding: utf-8 -*-

import torch
from torch import nn
from logger import get_root_logger
from .BALoss import BALoss
from .CharbonnierLoss import CharbonnierLoss
# from focal_frequency_loss import FocalFrequencyLoss as FFL
from .smooth_l1_loss import SmoothL1Loss

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.loss = []

        self.loss_module = nn.ModuleList()
        _logger = get_root_logger()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_func = nn.MSELoss(reduction='mean')
            elif loss_type == 'L1':
                loss_func = nn.L1Loss(reduction='mean')
            elif loss_type == 'Huber':
                loss_func = nn.HuberLoss(reduction='mean', delta=args.hdelta)
            elif loss_type == 'CharbonnierLoss':
                loss_func = CharbonnierLoss(eps=1e-12)
            elif loss_type == 'SmoothL1':
                loss_func = SmoothL1Loss(reduction='mean')
            elif loss_type == 'BALoss':
                loss_func = BALoss()
            # elif loss_type == 'FFL':
            #     loss_func = FFL(loss_weight=1.0, alpha=1.0)
            else:
                raise NotImplementedError
            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_func})

        for l in self.loss:
            if args.local_rank == 0:
                _logger.info('Loss Function: {:.3f} * {}'.format(l['weight'], l['type']))
            self.loss_module.append(l['function'])

        device = torch.device(args.device)
        self.loss_module.to(device)

    def forward(self, out, gt, mask=None):
        losses = []

        for i, l in enumerate(self.loss):
            if mask is None:
                loss = l['function'](out, gt)
            else:
                loss = l['function'](out* mask, gt* mask)
            effective_loss = l['weight'] * loss
            losses.append(effective_loss)
        return sum(losses)

