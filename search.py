# -*- coding: utf-8 -*-

import os
import time
import json
import torch
import random
import numpy as np
from data import get_loader
from dist import master_only
from predictor import Estimator
from utils import arch2key
from utils import update_summary, NpEncoder
from utils.base_lookup_table import get_model_efficiency, get_random_architecture

choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))


class EvolutionSearcher(object):

    def __init__(self, args, model, evaluate, logger, tb_logger, lookup_table):
        self.args = args
        self.epoch = 0
        self.vis_dict = {}

        self.memory = []
        self.candidates = []
        self.top_rmse = []
        self.keep_top_k = {args.select_num: [], 50: []}

        self.model = model
        self.logger = logger
        self.evaluate = evaluate
        self.tb_logger = tb_logger
        self.lookup_table = lookup_table

        self.cp_path = f'{args.log_path}checkpoints/{args.dataset}/{args.file_name}'

    @master_only
    def save_checkpoint(self):

        info = {'top_accuracies': self.top_rmse, 'memory': self.memory, 'candidates': self.candidates,
                'keep_top_k': self.keep_top_k, 'epoch': self.epoch}

        torch.save(info, f'{self.cp_path}/checkpoint-{self.epoch}.pth.tar')

        if os.path.exists(f'{self.cp_path}/last.pth.tar'):
            os.unlink(f'{self.cp_path}/last.pth.tar')

        os.link(f'{self.cp_path}/checkpoint-{self.epoch}.pth.tar', f'{self.cp_path}/last.pth.tar')
        self.logger.info(f'Save checkpoint to {self.cp_path}')

    @master_only
    def load_checkpoint(self):
        if not os.path.exists(f'{self.cp_path}/checkpoint-{self.epoch}.pth.tar'):
            return False
        info = torch.load(f'{self.cp_path}/checkpoint-{self.epoch}.pth.tar')
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        self.logger.info(f'Scheduled epochs: {self.epoch}')
        return True

    def is_legal(self, cand, calc_rmse=True):

        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]

        if 'visited' in info:
            return False
        model_efficiency = get_model_efficiency(cand, self.lookup_table, self.args)
        flops = model_efficiency['FLOPs']
        params = model_efficiency['Paras']
        latency = model_efficiency['Latency']
        if self.args.flop_max != -1:
            if flops > self.args.flop_max:
                self.logger.info('FLOPs limit exceed')
                return False
            info['complexity'] = flops
        if self.args.para_max != -1:
            if params > self.args.para_max:
                self.logger.info('parameters limit exceed {}'.format(params))
                return False
            info['complexity'] = params
        if self.args.latency_max != -1:
            if latency > self.args.latency_max:
                self.logger.info('Latency limit exceed')
                return False
            info['complexity'] = latency
        if calc_rmse:
            val_rmse = self.evaluate(self.model, get_loader(self.args, 'val'), 'val', self.args, cand)["RMSE/val"]
            info['RMSE'] = val_rmse
            info['visited'] = True
            info['arch'] = cand
        return True

    def get_rmse(self):
        for key in self.vis_dict.keys():
            cand = self.vis_dict[key]['arch']
            self.vis_dict[key]['RMSE'] = self.evaluate(
                self.model, get_loader(self.args, 'val'), 'val', self.args, cand)["RMSE/val"]

    def stack_random_cand(self, random_func, rand=False, batchsize=10):
        while True:
            if rand:
                cands = [
                    random_func(self.args.num_stages, self.args.num_blocks, self.args.num_base_ops,
                                self.args.num_fuse_ops)
                    for _ in range(batchsize)
                ]
            else:
                cands = [
                    random_func() for _ in range(batchsize)
                ]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
            for cand in cands:
                yield cand

    # def set_fixed(self, cand):
    #
    #     cand = list(cand)
    #
    #     if len(self.args.fix_rgb) > 0:
    #         fix_rgb = [int(i) for i in self.args.fix_rgb.split(',')]
    #         cand[: self.args.num_blocks * self.args.num_stages] = fix_rgb
    #
    #     if len(self.args.fix_dep) > 0:
    #         fix_dep = [int(i) for i in self.args.fix_dep.split(',')]
    #         cand[self.args.num_blocks * self.args.num_stages: self.args.num_blocks * self.args.num_stages * 2] = fix_dep
    #
    #     if len(self.args.fix_dec) > 0:
    #         fix_dec = [int(i) for i in self.args.fix_dec.split(',')]
    #         cand[self.args.num_blocks * self.args.num_stages * 2: self.args.num_blocks * self.args.num_stages * 3] = fix_dec
    #
    #     if len(self.args.fix_fusion) > 0:
    #         fix_fusion = [int(i) for i in self.args.fix_fusion.split(',')]
    #         cand[self.args.num_blocks * self.args.num_stages * 3:] = fix_fusion
    #
    #     return tuple(cand)

    def get_random(self, num):
        self.logger.info('Random Select ........')
        cand_iter = self.stack_random_cand(get_random_architecture, rand=True)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            # self.logger.info('Random {}/{}'.format(len(self.candidates), num))
        # self.logger.info('Random Num = {}'.format(len(self.candidates)))

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        # self.logger.info('Select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def get_mutation(self, k, mutation_num, m_prob, num_stages, num_base_ops, num_fuse_ops):
        assert k in self.keep_top_k
        # self.logger.info('Mutation ......')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            _cand = list(random.choice(self.keep_top_k[k]))
            for i in range(len(_cand)):
                if np.random.random_sample() < m_prob:
                    _cand[i] = np.random.randint(num_base_ops if i < len(_cand) - num_stages else num_fuse_ops)
            # print(num * 100 / len(_cand), '%')
            return tuple(_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            # self.logger.info(f'Mutation {len(res)}/{mutation_num}')

        # self.logger.info(f'Mutation Num {len(res)}')
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        # self.logger.info('Crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            # self.logger.info('Crossover {}/{}'.format(len(res), crossover_num))

        # self.logger.info('Crossover_num = {}'.format(len(res)))
        return res

    def init_archs(self, num_arch):
        cand_archs = []
        cand_iter = self.stack_random_cand(get_random_architecture, rand=True)
        while len(cand_archs) < 1000:
            cand = next(cand_iter)
            if not self.is_legal(cand, calc_rmse=False):
                continue
            cand_archs.append(cand)
            # print(len(cand_archs))

        estimator = Estimator(
            num_base_ops=self.args.num_base_ops, num_fuse_ops=self.args.num_fuse_ops,
            embed_size=32, hidden_size=32, num_layers=2, lr=1e-4, device=self.args.device,
        )
        if os.path.exists(self.args.model_path):
            if self.args.local_rank == 0:
                estimator.load(self.args.model_path)
                print('Performance Estimator Loaded ...')

        res = []
        for arch in cand_archs:
            res.append(estimator.predict(torch.LongTensor(np.array(arch).reshape(1, -1)).cuda()))
        res = torch.stack(res).squeeze()
        topk_indices = torch.argsort(res)[:num_arch]
        topk_cands = [cand_archs[i] for i in topk_indices]

        for cand in topk_cands:
            self.is_legal(cand, calc_rmse=True)
            self.candidates.append(cand)

    def search(self):
        reverse = False

        if self.args.dataset.lower() in ['wv2', 'gf2', 'nir']:
            reverse = True
        if self.args.local_rank == 0:
            self.logger.info(
                f'Population Num = {self.args.population_num}, Select Num = {self.args.select_num}, '
                f'Mutation Num = {self.args.mutation_num}, Crossover Num = {self.args.crossover_num}, '
                f'Random Num = {self.args.population_num - self.args.mutation_num - self.args.crossover_num}'
            )
        # self.load_checkpoint()
        if self.args.init_arch:
            self.init_archs(self.args.population_num)
        else:
            self.get_random(self.args.population_num)

        # self.logger.info(self.candidates)
        start_time = time.time()
        for epoch in range(self.epoch, self.args.epochs):
            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)      # 所有遍历过得arch

            self.update_top_k(
                self.candidates, k=self.args.select_num, key=lambda x: self.vis_dict[x]['RMSE'], reverse=reverse)
            self.update_top_k(self.candidates, k=50, key=lambda x: self.vis_dict[x]['RMSE'], reverse=reverse)

            # self.logger.info(f'epoch = {epoch} : top {len(self.keep_top_k[50])} result')
            tmp_rmse = []
            for i, cand in enumerate(self.keep_top_k[50]):
                # self.logger.info('No.{} {} Val RMSE = {}, complexity = {}'.format(
                #     i + 1, cand, self.vis_dict[cand]['RMSE'], self.vis_dict[cand]['complexity']))
                tmp_rmse.append(self.vis_dict[cand]['RMSE'])
            self.top_rmse.append(tmp_rmse)

            sel_rmse = [self.vis_dict[cand]['RMSE'] for cand in self.keep_top_k[self.args.select_num]]
            if self.args.random_search:
                # mutation = self.get_mutation(0, 0, 0,
                #                              self.args.num_stages,
                #                              self.args.num_base_ops, self.args.num_fuse_ops)
                #
                # crossover = self.get_crossover(0, 0)

                self.candidates = []
                if len(self.candidates) != 0:
                    raise RuntimeError
            else:
                mutation = self.get_mutation(self.args.select_num, self.args.mutation_num, self.args.m_prob,
                                             self.args.num_stages,
                                             self.args.num_base_ops, self.args.num_fuse_ops)

                crossover = self.get_crossover(self.args.select_num, self.args.crossover_num)
                self.candidates = mutation + crossover
            self.get_random(self.args.population_num)
            self.epoch = epoch
            if self.epoch % 1 == 0:
                self.save_checkpoint()
            log_status = {
                'Selected_RMSE': np.mean(sel_rmse), 'Top_50_RMSE': np.mean(tmp_rmse),
                'Visited Cands': np.sum([len(i) for i in self.memory]), 'Best RMSE': sel_rmse[0]
            }
            self.logger.info('Best RMSE {}, Visited Cands'.format(sel_rmse[0], np.sum([len(i) for i in self.memory])))
            update_summary(epoch, start_time, log_status, self.tb_logger)

        save_dict = {}
        for key in self.vis_dict.keys():
            save_key = ''.join([str(i) for i in key])
            save_dict[save_key] = self.vis_dict[key]
        with open(f'{self.cp_path}/vis_dict.json', 'w') as f:
            f.write(json.dumps(save_dict, cls=NpEncoder))


if __name__ == '__main__':
    pass
