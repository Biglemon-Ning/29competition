'''
This file is to display the hyperparameters of the method.
Author : Yi-min Fu, Liang-bo Ning, Han-rui Shi, Shu-qian Zhou, Rui Liu 
Institution : Northwestern Polytechnical University
Date : 2022.7.18
'''

batch_size = 32
epochs = 30
lr = 0.01 # 0.0000001
momentum = 0.9
seed = 8   #8
log_interval = 20
l2_decay = 5e-4
cuda = True
seed = 8
accuracy_max = 0.912

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}