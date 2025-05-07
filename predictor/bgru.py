# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from predictor.pdata import ArchData
from predictor.rankloss import RankNet
from torch.utils.data import DataLoader

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, scale=2, bias=False, batch_first=True, bidirectional=True):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.scale = int(scale * hidden_size)
        self.num_direction = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_direction, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs):

        h0 = torch.zeros(self.num_layers * self.num_direction , inputs.size(0), self.hidden_size)
        h0 = h0.cuda() if inputs.get_device() != -1 else h0
        _, h0 = self.gru(inputs, h0)
        if self.num_direction == 2:
            h0 = h0.permute(1, 0, 2)
            h0 = torch.cat((h0[:, 0], h0[:, 1]), dim=1)

        return self.fc(h0)


class Net(nn.Module):
    def __init__(self, num_base_ops, num_fuse_ops, embed_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.base_embedding = nn.Embedding(num_base_ops, embed_size)
        self.fuse_embedding = nn.Embedding(num_fuse_ops, embed_size)
        self.model = GRU(embed_size, hidden_size, num_layers)

    def forward(self, x):
        base_feat = self.base_embedding(x[:, :-4])
        fuse_feat = self.fuse_embedding(x[:, -4:])
        combined = torch.cat([base_feat, fuse_feat], dim=1)
        return self.model(combined)


class Estimator:
    def __init__(self, num_base_ops, num_fuse_ops, embed_size=32, hidden_size=32,
                 num_layers=2, lr=1e-3, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = lr
        self.model = Net(num_base_ops, num_fuse_ops, embed_size, hidden_size, num_layers).to(self.device)
        self.criterion = RankNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, dataloader, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {total_loss / len(dataloader):.4f}")

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[Eval] Average Loss: {avg_loss:.4f}")
        return avg_loss

    def fine_tune(self, new_dataloader, num_epochs=5):
        self.lr = self.lr * 0.9
        self.train(new_dataloader, num_epochs)
        print(f"[Fine-tune] lr: {self.lr}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        # print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        # print(f"Model loaded from {path}")


if __name__ == '__main__':
    from config import args

    batch_size = 16
    num_epochs = 100
    embed_size = 32
    hidden_size = 32
    num_layers = 2
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = args.train_arch_path.replace('/predictor', '')

    train_loader = DataLoader(ArchData(file_path=file_path), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ArchData(file_path=file_path, split='val'), batch_size=batch_size)

    estimator = Estimator(num_base_ops=args.num_base_ops, num_fuse_ops=args.num_fuse_ops,
                          embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                          lr=lr, device=device)

    estimator.train(train_loader, num_epochs=num_epochs)
    estimator.save(path=args.model_path)
    print('==> Model saved to {}'.format(args.model_path))