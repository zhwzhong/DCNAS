# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F

# class RankLoss(torch.nn.Module):
#     def __init__(self, sigma=1):
#         super(RankLoss, self).__init__()
#         self.sigma = sigma
#
#     def forward(self, score_predict, score_real):
#         """
#         Calculate the loss of RankNet
#         :param score_predict: 1xN tensor with model output score
#         :param score_real: 1xN tensor with real score (越大表示网络越好)
#         :return:
#         """
#         score_diff = self.sigma * (score_predict.t() - score_predict)
#         score_diff = torch.sigmoid(score_diff)
#         tij = (1.0 + torch.sign(score_real.t() - score_real)) / 2.0
#         loss_mat = tij * torch.log(score_diff) + (1 - tij) * torch.log(1 - score_diff)
#         return -torch.mean(loss_mat)


def get_pairwise_comp_probs(batch_preds, batch_std_labels, sign=False, sigma=None):
    '''
    Get the predicted and standard probabilities p_ij which denotes d_i beats d_j
    @param batch_preds:
    @param batch_std_labels:
    @param sigma:
    @param sign: sign function
    '''
    # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
    batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
    batch_p_ij = torch.sigmoid(sigma * batch_s_ij)
    # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
    # ensuring S_{ij} \in {-1, 0, 1}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
    if sign:
        batch_std_p_ij = 0.5 * (1.0 + torch.sign(batch_Sij))
    else:
        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    return batch_p_ij, batch_std_p_ij

class RankNet(torch.nn.Module):
    '''
    Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
    Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
    '''
    def __init__(self, sign=False, sigma=1.0):
        super(RankNet, self).__init__()
        self.sigma = sigma
        self.sign = sign
    def forward(self, batch_preds, batch_std_labels):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        batch_preds = batch_preds.reshape(1, -1)
        batch_std_labels = batch_std_labels.reshape(1, -1)
        batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(
            batch_preds=batch_preds, batch_std_labels=batch_std_labels, sign=self.sign, sigma=self.sigma)

        _batch_loss = F.binary_cross_entropy(
            input=torch.triu(batch_p_ij, diagonal=1), target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        batch_loss = torch.mean(torch.mean(_batch_loss, dim=(2, 1)))
        return batch_loss
