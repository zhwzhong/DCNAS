# -*- coding: utf-8 -*-

import torch.optim as optim

def make_optimizer(args, targets):

    if args.opt == 'AMSGrad':
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    else:
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer