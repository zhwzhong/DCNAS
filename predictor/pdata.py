# -*- coding: utf-8 -*-

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split



class ArchData(Dataset):
    def __init__(self, file_path, split='train', test_size=0.2, random_seed=42):
        data = pd.read_csv(file_path, header=None, skiprows=1)

        inputs = data[0].str.split('_', expand=True)
        inputs = inputs.apply(pd.to_numeric, errors='coerce').fillna(0)
        targets = pd.to_numeric(data[1], errors='coerce').fillna(0)

        inputs_np = inputs.values.astype('int64')
        targets_np = targets.values.astype('float32')

        # 划分 train/val
        X_train, X_val, y_train, y_val = train_test_split(
            inputs_np, targets_np, test_size=test_size, random_state=random_seed)

        if split == 'train':
            self.inputs = torch.tensor(X_train)
            self.targets = torch.tensor(y_train)
        elif split == 'val':
            self.inputs = torch.tensor(X_val)
            self.targets = torch.tensor(y_val)
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.LongTensor(self.inputs[idx]), self.targets[idx]

