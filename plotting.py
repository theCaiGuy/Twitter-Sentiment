import random
import os
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt

PATH_1 = './6b_data'
PATH_2 = './twitter27b_data'
PATH_3 = './deeper_lr_decay_data'
PATH_4 = './deeper_l2_data'
PATH_5 = './deeper_l2_no_decay_2'
PATH_6 = './deeper_data'
PATH_7 = './deeper_l2_lr_decay_data'
PATH_8 = './twitter_l2_Data'

data_1_train = open(PATH_8 + '/train_loss.txt', 'r')
data_1_val = open(PATH_8 + '/val_loss.txt', 'r')

train_iters = []
train_loss = []
train_acc = []
train_f1 = []

dev_iters = []
dev_loss = []
dev_acc = []
dev_f1 = []

for line in data_1_train:
    line_toks = line.split()
    train_iters.append(int(line_toks[1]))
    train_loss.append(float(line_toks[3]))
    train_acc.append(float(line_toks[5]))
    train_f1.append(float(line_toks[7]))

max_dev_f1 = 0.0
max_dev_f1_epoch = 0
max_dev_acc_epoch = 0
max_dev_acc = 0.0

for line in data_1_val:
    line_toks = line.split()
    dev_iters.append(int(line_toks[3]))
    dev_loss.append(float(line_toks[5]))
    dev_acc.append(float(line_toks[7]))
    dev_f1.append(float(line_toks[9]))
    if float(line_toks[9]) > max_dev_f1:
        max_dev_f1 = float(line_toks[9])
        max_dev_f1_epoch = int(line_toks[1])
    if float(line_toks[7]) > max_dev_acc:
        max_dev_acc = float(line_toks[7])
        max_dev_acc_epoch = int(line_toks[1])

print ('max f1 epoch')
print (max_dev_f1_epoch)
print ('max f1')
print (max_dev_f1)

print ('max acc epoch')
print (max_dev_acc_epoch)
print ('max acc')
print (max_dev_acc)


plt.plot(train_iters, train_f1, 'b', dev_iters, dev_f1, 'r')
plt.show()
