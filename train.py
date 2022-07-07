'''
This file is used to train the network.
Author : Lemon
Institution : Northwestern Polytechnical University
Data : 2022.7.7
'''
import os
import torch
import math
import data_loader
import numpy as np 
import torch.nn.functional as F
from torch.autograd import Variable
from Config import *
from npu_model import NPU_model

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(seed)
else:
    device = torch.device("cpu")
    torch.manual_seed(seed)

root = os.getcwd()
train_path = r'/train'
val_path = r'/validation'

def get_data(root, path):
    loader = data_loader.load_training(root, path, batch_size, kwargs)
    return loader

def train():
    model = NPU_model(10).to(device)
    model.train()

    epoch = 0
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    optimizer = torch.optim.SGD([
        {'params':model.resnet.parameters(), 'lr':LEARNING_RATE / 10},
        {'params':model.dense.parameters(), 'lr':LEARNING_RATE},
    ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    train_loader = get_data(root, train_path)
    val_loader = get_data(root, val_path)

    for data, label in train_loader:
        epoch += 1
        data, label = data.to(device), label.to(device)
        data, label = Variable(data), Variable(label)

        pred = model(data)
        loss = F.nll_loss(F.log_softmax(pred, dim=1), label)
        loss.backward()
        optimizer.step()
        print(f'The loss is : {loss}')

        if epoch % log_interval == 0:
            val(model, val_loader)

def val(model, data_loader):
    model.eval()
    accuracy_max = 0
    correct_num = 0
    accuracy = 0
    for i, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        data, label = Variable(data), Variable(label)

        pred = model(data).max(1)[1]
        equal = torch.eq(pred, label)
        correct_num += equal.sum()
        rate = round(float(i / len(data_loader) * 100), 2)
        print("\r", f'Start to evaluate : {rate}%', "▓" * (int(rate) // 2), end="", flush=True)
    accuracy = correct_num / len(data_loader.dataset)
    if accuracy > accuracy_max:
        accuracy_max = accuracy
    print(f'The current accuracy is {accuracy}, the max accuracy is {accuracy_max}')

if __name__ == '__main__':
    train()


    