'''
This file is to display the hyperparameters of the method.
Author : Lemon
Institution : Northwestern Polytechnical University
Data : 2022.7.7
'''

batch_size = 256
epochs = 2000
lr = 0.0000001
momentum = 0.9
seed = 8   #8
log_interval = 20
l2_decay = 5e-4
cuda = True
seed = 8
accuracy_max = 0

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}