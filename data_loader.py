'''
This file is used to load the training data.
Author : Lemon
Institution : Northwestern Polytechnical University
Data : 2022.7.13
'''

import os 
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def load_training(root_path, dir, batch_size, kwargs):
    data = My_datasets(root=root_path + dir)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    data = My_datasets(root=root_path + dir)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader

class My_datasets(Dataset):
    def __init__(self, root):
        self.root = root
        self.dir = os.listdir(self.root)
        self.data_list = []
        self.label = []
        self.class_to_idx = {}
        for i in range(len(self.dir)):
            self.data_list.extend(os.listdir(os.path.join(self.root, self.dir[i])))
            self.label.extend((i * np.ones(len(os.listdir(os.path.join(self.root, self.dir[i]))))).astype(np.int).tolist())
            self.class_to_idx[self.dir[i]] = i

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_name = self.data_list[index]
        data_path = os.path.join(self.root, self.dir[self.label[index]], data_name)
        data = self.data_convert(data_path)
        label = self.label[index]
        data = self.data_transform(data)
        return data, label

    def data_convert(self, path):
        with open(path, 'r') as file:
            data = file.readlines()
            data = [float(i.strip('\n')) for i in data]
            data = torch.tensor(list(data))
        return data

    def data_transform(self, data):
        mean = data.mean()
        std = data.std()
        data = (data - mean) / std
        return data