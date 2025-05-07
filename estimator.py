# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from config import args
from predictor import Estimator
import torch.optim as optim
from predictor.rankloss import RankNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

def key2int(key):
    return np.array([int(k) for k in key.split('_')])

class ArchData(Dataset):
    def __init__(self, attr):
        super(ArchData, self).__init__()
        archs = pd.read_csv(args.train_arch_path)
        X, Y = list(archs.loc[:, "Unnamed: 0"]), list(archs.loc[:, "0"])

        X = [key2int(x) for x in X]
        num_train = int(len(X) * 0.8)
        Y = 1 - (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

        if attr == 'train':
            self.x, self.y = X[num_train:], Y[num_train:]
        else:
            self.x, self.y = X[: num_train], Y[: num_train]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = torch.LongTensor(self.x[item])
        y = torch.FloatTensor(np.array(self.y[item]))
        return x, y



def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = []
    for batch_idx, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, target = inputs.to(device), target.to(device)
        out = model(inputs)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    for (inputs, target) in val_loader:
        inputs, target = inputs.to(device), target.to(device)
        out = model(inputs)
        val_loss.append(criterion(out, target).item())
    return np.mean(val_loss)

def main():
    device = torch.device("cuda")
    model = Estimator(args, embed_size=32, hidden_size=32, num_layers=2).to(device)

    criterion = RankNet(sign=args.rank_sign)

    optimizer = optim.Adam(model.parameters(), lr=args.estimator_lr)
    scheduler = MultiStepLR(optimizer, milestones=[500, 800], gamma=0.1)
    train_data = DataLoader(ArchData('train'), batch_size=64, shuffle=True)
    val_data = DataLoader(ArchData('val'), batch_size=4, shuffle=False, drop_last=False)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_data, optimizer, criterion, device)

        if epoch % 200 == 0:
            print('===> TrainLoss', loss)
            val_loss = evaluate(model, val_data, criterion, device)
            print('===> Validation Loss', val_loss)
        scheduler.step()

    torch.save(model.state_dict(), args.model_path)

if __name__ == '__main__':

    main()

