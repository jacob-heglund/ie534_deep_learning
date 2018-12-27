'''
Filename: c:\home\classes\IE534_DL\hw5\src\hw5.py
Path: c:\home\classes\IE534_DL\hw5\src
Created Date: Wednesday, October 10th 2018, 3:53:26 pm
Author: Jacob Heglund

Copyright (c) 2018 Jacob Heglund
'''

##########################################
import os
import sys
import numpy as np
from numpy.linalg import norm
from numpy import shape
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  PIL import Image
import pandas as pd
from random import randint
import json
from tiny_imagenet_loader import tinyImageNetDataset

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models
##########################################
# program controls
blue_waters = 1

train_batch_size = 15
val_batch_size = 100
##########################################
if blue_waters:
    src_dir = '/mnt/c/scratch/training/tra392/hw5/src/'

else:
    src_dir = 'C:/home/classes/IE534_DL/hw5/src'

train_data_dir = 'data/tiny-imagenet-200/train/'
val_data_dir = 'data/tiny-imagenet-200/val'
train_annotations_path = os.path.join(src_dir, 'img_df.csv')
val_annotations_path = os.path.join(src_dir, 'data/tiny-imagenet-200/val/val_annotations.txt')
backup_dir = os.path.join(src_dir, 'backup')

##########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters
num_epochs = 45
learning_rate = 0.001
# decay learning rate by a factor of 'scaling' every 'decay_epoch' epochs
decay_epoch = 10
scaling = 10

# should reach 50% after 10 epochs or so
model = models.resnet50(pretrained = True)
#model = models.resnet101(pretrained = True)
#model = models.resnet18(pretrained = True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.to(device)

# Loss function
triplet_loss = nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# initialize a datafile for storing loss after each epoch
# for each epoch, it should be loss += loss.item()
tmp_loss = []
loss_fn = 'loss.txt'
loss_path = os.path.join(backup_dir, loss_fn)
with open(loss_path, 'w') as f:
    json.dump(tmp_loss, f)

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

##########################################
# CIFAR100 dataset info
# image size: 64x64 pixels
# number classes: 200
# images per class: 500
# training: 100,000
# testing: 10,000

# Image preprocessing modules
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])

val_transform =transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])

# TinyImageNet Datasets
train_dataset = tinyImageNetDataset(src_dir = src_dir, data_dir = train_data_dir, annotations_path = train_annotations_path, train = True,transform = train_transform)

val_dataset = tinyImageNetDataset(src_dir = src_dir, data_dir = val_data_dir, annotations_path = val_annotations_path, train = False, transform = val_transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle=True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = val_batch_size, shuffle = True, num_workers = 0)

############################################
def lrSchedule(epoch, decay_epoch, scaling, optimizer, learning_rate):    
    if (epoch+1) % decay_epoch == 0:
        learning_rate /= 10    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


# training (in words)
'''
- done-send a triplet (or batch of triplets) through a network to get feature maps
- done-store all the feature maps in memory for use during validation
- done-calculate the loss with a gap parameter set to 1.0
- done-backprop
- done-track loss for each epoch
'''

def train(learning_rate):
    print('\n------------Training------------\n')
    class_buffer = []

    all_maps = torch.zeros([1, 2048, 1, 1])
    first = True

    time1 = time.time()
    num_steps = len(train_loader)
    
    for i, loaded_data in enumerate(train_loader):
        query_imgs = loaded_data[0][0]
        pos_imgs = loaded_data[0][1]
        neg_imgs = loaded_data[0][2]
        query_class = loaded_data[1]
        query_img_path = loaded_data[2]

        query_imgs = query_imgs.to(device)
        pos_imgs = pos_imgs.to(device)
        neg_imgs = neg_imgs.to(device)

        # forward pass
        q_map = model(query_imgs)
        p_map = model(pos_imgs)
        n_map = model(neg_imgs)
        loss = triplet_loss(q_map, p_map, n_map)        
        
        
        # zero any stored gradients
        optimizer.zero_grad()
        # calculate new gradients
        loss.backward()
        # avoid numerical issues
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        
        optimizer.step()
        
        q_map_cpu = q_map.to('cpu')
        if first:
            all_maps = torch.cat((all_maps, q_map_cpu), 0)
            all_maps = all_maps[1:, :]
            first = 0
        else:
            all_maps = torch.cat((all_maps, q_map_cpu), 0)

        # class_buffer[i] = [query_class (string), query_img_path (str)]
        for j in range(train_batch_size):
            query_class_path = [query_class[j], query_img_path[j]]
            class_buffer.append(query_class_path)

        if (i+1) % 100 == 0:
            percent_complete = (float(i+1) / (float(num_steps)))*100.
            print ("Epoch [{}/{}]-----Step [{}/{}]-----Percent Complete: {}-----Loss: {:.4f}".format(epoch, num_epochs, i+1, num_steps, percent_complete, loss.item()))
            
    if (epoch+1) % 20 == 0:
        learning_rate /= 10

    time2 = time.time()
    print('Epoch Runtime: {} seconds'.format(time2-time1))
    all_maps_cpu = all_maps.to('cpu')
    epoch_loss = loss.item()
    return epoch_loss, all_maps_cpu, class_buffer

##########################################
# saves class buffer, loss, model parameters to disk
def save_epoch_data(backup_dir, epoch_loss, all_maps_cpu, class_buffer):
    print('\n------------Saving Data------------\n')
    epoch_dir = 'epoch_' + str(epoch)
    save_path = os.path.join(backup_dir, epoch_dir)
    os.makedirs(save_path)
    # save model to a unique file for each epoch
    model_fn = 'epoch_' + str(epoch) + '_resnet.ckpt'
    model_path = os.path.join(save_path, model_fn)
    torch.save(model.state_dict(), model_path)  
    
    # initialize a datafile for storing class na`mes 
    class_buffer_fn = 'epoch_' + str(epoch) + '_class_buffer.txt'
    class_path = os.path.join(save_path, class_buffer_fn)
    with open(class_path, 'w') as f:
        json.dump(class_buffer, f)

    # save all_maps to file
    all_maps_fn = 'epoch_' + str(epoch) + '_all_maps.txt'
    all_maps_path = os.path.join(save_path, all_maps_fn)
    with open(all_maps_path, 'wb') as f:
        torch.save(all_maps_cpu, f)

    # load loss, add to the list, save to file
    with open(loss_path, 'r') as f:
        tmp_loss = json.load(f)
    tmp_loss.append(epoch_loss)
    with open(loss_path, 'w') as f:
        json.dump(tmp_loss, f)

if __name__ == '__main__':
    for epoch in range(num_epochs):
        epoch_loss, all_maps_cpu, class_buffer = train(learning_rate = learning_rate)
        save_epoch_data(backup_dir, epoch_loss, all_maps_cpu, class_buffer)




