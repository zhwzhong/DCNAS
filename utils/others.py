# -*- coding: utf-8 -*-

import tqdm
import data
import numpy as np
from config import args
from models.guided_filter import GuidedFilter
from utils import normalize, de_normalize, metrics
from torchvision.transforms.functional import rgb_to_grayscale

GF = GuidedFilter(r=4)

def get_bic_rmse():
    gf = GuidedFilter(r=4)
    args.val_batch_size = 8
    args.distributed = False
    save_result = True
    for test_name in ['nyu', 'lu', 'middlebury', 'sintel', 'didoe', 'sun', 'diml', 'rd']:
        rmse_values = ''
        for scale in [4, 8, 16]:
            sum_rmse = []
            args.scale  = scale
            sv_path = '/data/zhwzhong/PycharmProjects/GDSR/results/NYU/bicubic'
            if save_result:
                create_dir(f"{sv_path}/GT/{scale}/{test_name}")
                create_dir(f"{sv_path}/LR/{scale}/{test_name}")
                create_dir(f"{sv_path}/GF/{scale}/{test_name}")
                create_dir(f"{sv_path}/RGB/{scale}/{test_name}")
            val_data = data.get_loader(args, test_name)
            for _, sample in enumerate(val_data):

                lr_up, gt_img = sample['lr_up'], sample['img_gt']
                lr_up = normalize(lr_up, gt_img=gt_img, attr=test_name, args=args)
                gf_out = gf(rgb_to_grayscale(sample['img_rgb'], 1), lr_up)

                lr_up = de_normalize(lr_up, gt_img=gt_img, attr=test_name, args=args)
                gf_out = de_normalize(gf_out, gt_img=gt_img, attr=test_name, args=args)

                if save_result:
                    img_name = sample['img_name']
                    for i in range(sample['img_gt'].size(0)):
                        print(f"Image Saved to {sv_path}/GF/{scale}/{test_name}/{img_name[i]}.npy")
                        np.save(f"{sv_path}/GF/{scale}/{test_name}/{img_name[i]}",
                                gf_out[i].squeeze().detach().cpu().numpy())

                        np.save(f"{sv_path}/GT/{scale}/{test_name}/{img_name[i]}.npy",
                                gt_img[i].squeeze().detach().cpu().numpy())

                        np.save(f"{sv_path}/LR/{scale}/{test_name}/{img_name[i]}",
                                lr_up[i].squeeze().detach().cpu().numpy())

                        np.save(f"{sv_path}/RGB/{scale}/{test_name}/{img_name[i]}.npy",
                                sample['img_rgb'][i].squeeze().detach().cpu().numpy())

                sum_rmse.append(metrics(gf_out, gt_img, sample['img_mask'], args.gdata, test_name)['RMSE'] * lr_up.size(1))
            sum_rmse = round(np.sum(sum_rmse) / len(sum_rmse), 2)
            rmse_values += '{} &'.format(sum_rmse)
        print(test_name, rmse_values)


def d_data_rmse():
    args.gdata = True
    args.val_batch_size = 16
    args.distributed = False
    for dataset in ['NYU', 'DIML', 'Middlebury']:
        rmse_values = ''
        args.dataset = dataset
        for scale in [4, 8, 16]:
            sum_rmse = []
            args.scale = scale
            val_data = data.get_loader(args, 'test')
            for _, sample in enumerate(tqdm.tqdm(val_data, desc=dataset)):
                lr_up, gt_img = sample['lr_up'], sample['img_gt']
                lr_up = normalize(lr_up, gt_img=gt_img, attr=args.dataset, args=args)
                lr_up = de_normalize(lr_up, gt_img=gt_img, attr=args.dataset, args=args)
                for i in range(lr_up.size(0)):
                    sum_rmse.append(metrics(lr_up[i: i + 1], gt_img[i: i + 1], sample['img_mask'],
                                                   args.gdata, args.dataset)[ 'RMSE'])
            sum_rmse = round(np.sum(sum_rmse) / len(sum_rmse), 2)
            rmse_values += '{} &'.format(sum_rmse)
        print(args.dataset, rmse_values)

def rd_data():
    args.scale = 2
    args.dataset = 'RD'
    args.val_batch_size = 8

    gf = GuidedFilter(r=4)
    train_data = data.get_loader(args, 'rd')
    gf_rmse = []
    sum_rmse = []
    for _, sample in enumerate(train_data):
        print(sample['img_lr'].size(), sample['img_rgb'].size())
        lr_up, gt_img = sample['lr_up'], sample['img_gt']
        lr_up = normalize(lr_up, gt_img=gt_img, attr=args.dataset, args=args)
        gf_out = gf(rgb_to_grayscale(sample['img_rgb'], 1), lr_up)
        gf_out = de_normalize(gf_out, gt_img=gt_img, attr='rd', args=args)
        lr_up = de_normalize(lr_up, gt_img=gt_img, attr=args.dataset, args=args)
        # print(lr_up.max(), lr_up.size())
        for i in range(lr_up.size(0)):
            gf_rmse.append(metrics(gf_out[i: i + 1], gt_img[i: i + 1], sample['img_mask'], args.gdata, args.dataset)['RMSE'])
            sum_rmse.append(metrics(lr_up[i: i + 1], gt_img[i: i + 1], sample['img_mask'], args.gdata, args.dataset)['RMSE'])
    # sum_rmse = round(np.sum(sum_rmse) / len(sum_rmse), 2)
    print('%.5f' % (np.sum(sum_rmse) / len(sum_rmse)))
    print('%.5f' % (np.sum(gf_rmse) / len(sum_rmse)))


if __name__ == '__main__':
    # args.dataset = 'RD'
    # args.scale = 2
    # args.patch_size = 32
    # val_data = data.get_loader(args, 'train')
    # for _, sample in enumerate(val_data):
    #     print(sample['lr_up'].shape)
    rd_data()
    # get_bic_rmse()
    # d_data_rmse()