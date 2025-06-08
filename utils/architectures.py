# -*- coding: utf-8 -*-

import numpy as np


def arch2key(arch, num_stages, num_blocks):
    key = []
    for architecture in ['rgb_encoder', 'dep_encoder', 'rec_decoder']:
        for i in range(num_stages):
            for j in range(num_blocks):
                key.append(arch[architecture][i][j])
    key.extend(list(arch['fuse_op']))
    return tuple(key)


def key2arch(key, num_stages, num_blocks):
    arch = {}
    start_index = 0
    for i, architecture in enumerate(['rgb_encoder', 'dep_encoder', 'rec_decoder']):
        arch[architecture] = []
        for j in range(num_stages):
            s_idx = start_index + j * num_blocks
            arch[architecture].append(np.array(key[s_idx: s_idx+ num_blocks]))
        start_index += num_blocks * num_stages
    arch['fuse_op'] = np.array(key[len(key) - num_stages:])

    assert key == arch2key(arch, num_stages, num_blocks), 'Architecture to key error...'
    return arch

def random_mb(num_blocks, num_base_ops):
    nb = np.random.randint(0, num_blocks)
    sel_ops = list(np.random.randint(0, num_base_ops, nb))
    while len(sel_ops) != num_blocks:
        sel_ops.append(0)
    return np.array(sel_ops)


def get_random_architecture(num_stages, num_blocks, num_base_ops, num_fuse_ops):

    rgb_encoder = [np.random.randint(0, num_base_ops, num_blocks) for _ in range(num_stages)]
    dep_encoder = [np.random.randint(0, num_base_ops, num_blocks) for _ in range(num_stages)]
    rec_decoder = [np.random.randint(0, num_base_ops, num_blocks) for _ in range(num_stages)]
    fuse_operator = np.random.randint(0, num_fuse_ops, num_stages)

    arch = {
        'rgb_encoder': rgb_encoder, 'dep_encoder': dep_encoder, 'rec_decoder': rec_decoder, 'fuse_op': fuse_operator
    }
    return arch2key(arch, num_stages, num_blocks)


def get_fix_architecture(num_stages, num_blocks):
    return arch2key({
        'rgb_encoder': [np.array([0, 3]), np.array([0, 2]), np.array([6, 3]), np.array([0, 4])],
        'dep_encoder': [np.array([1, 3]), np.array([4, 2]), np.array([6, 6]), np.array([2, 0])],
        'rec_decoder': [np.array([6, 1]), np.array([1, 4]), np.array([1, 4]), np.array([6, 4])],
        'fuse_op': np.array([6, 6, 3, 6])
    }, num_stages, num_blocks)

