'''
This file is to display the hyperparameters of the method.
Author : Lemon
Institution : Northwestern Polytechnical University
Data : 2022.7.7
'''

batch_size = 32
epochs = 200
lr = 0.01
momentum = 0.9
seed = 8   #8
log_interval = 10
l2_decay = 5e-4
cuda = True
seed = 8

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}