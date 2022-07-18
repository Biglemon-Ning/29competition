'''
This file is used to train and test the model.
Author : Yi-min Fu, Liang-bo Ning, Han-rui Shi, Shu-qian Zhou, Rui Liu 
Institution : Northwestern Polytechnical University
Date : 2022.7.18
'''
import os
import torch
import math
import argparse
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

paras = argparse.ArgumentParser()
paras.add_argument('-s', '--status', type=str, default='train', help='This parameter is to decide whether it will train the model or not')
args = paras.parse_args()

root = os.getcwd()
train_path = r'/train'
val_path = r'/validation'

train_loader = data_loader.load_training(root, train_path, batch_size, kwargs)
val_loader = data_loader.load_training(root, val_path, batch_size, kwargs)

def train():
    model = NPU_model(10).to(device)
    model.train()

    epoch = 0
    LEARNING_RATE = lr / math.pow((1 + 10 * epoch / 20), 0.75)
    optimizer = torch.optim.SGD([
        {'params':model.resnet.parameters(), 'lr':LEARNING_RATE / 10},
        {'params':model.dense.parameters(), 'lr':LEARNING_RATE},
    ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    for i in range(epochs):
        epoch += 1
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            data, label = Variable(data), Variable(label)

            optimizer.zero_grad()
            pred = model(data)
            loss = F.nll_loss(F.log_softmax(pred, dim=1), label)
            loss.backward()
            optimizer.step()
            print(f'{i} The loss is : {loss}')

        val(model, val_loader)

def val(model, data_loader):
    model.eval()
    correct_num = 0
    accuracy = 0
    result = []
    global accuracy_max
    if args.status == 'train':
        for i, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            data, label = Variable(data), Variable(label)

            pred = model(data).max(1)[1]
            result.extend(pred.tolist())
            equal = torch.eq(pred, label)
            correct_num += equal.sum()
            rate = round(float(i / len(data_loader) * 100), 2)
            print("\r", f'Start to evaluate : {rate}%', "▓" * (int(rate) // 2), end="", flush=True)

        accuracy = correct_num / len(data_loader.dataset)
    
        if args.status == 'train':
            if accuracy > accuracy_max:
                accuracy_max = accuracy
                print('\n Start to save model.')
                torch.save(model.state_dict(), os.path.join(root, 'result_model.pth'))
            print(f'\n The current accuracy is {accuracy}, the max accuracy is {accuracy_max}')

    elif args.status == 'test':
        for i, data in enumerate(data_loader):
            data = data.to(device)
            data = Variable(data)

            pred = model(data).max(1)[1]
            result.extend(pred.tolist())
            rate = round(float(i / len(data_loader) * 100), 2)
            print("\r", f'Start to evaluate : {rate}%', "▓" * (int(rate) // 2), end="", flush=True)
        print('\n The evaluation process is finished.')
        with open(os.path.join(root, 'output.txt'), 'w') as file:
            for i in range(len(result)):
                file.writelines([str(result[i]), '\n'])
                

if __name__ == '__main__':
    if args.status == 'train':
        train()
    elif args.status == 'test':
        test_loader = data_loader.load_testing(root, r'/test', batch_size, kwargs)
        model = NPU_model(10).to(device)
        model.load_state_dict(torch.load(os.path.join(root, 'result_model.pth'), map_location=device))
        val(model, test_loader)



    