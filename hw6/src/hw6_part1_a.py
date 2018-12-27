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
train_batch_size = 128
val_batch_size = 128
# set to the value of the epoch checkpoint you want to load, otherwise set to 0
load_epoch = 0
##########################################
# set important directories
if blue_waters:
    src_dir = '/mnt/c/scratch/training/tra392/hw6/src'
else:
    src_dir = 'C:/home/classes/IE534_DL/hw6/src'

ckpt_dir = os.path.join(src_dir, 'checkpoints')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

data_dir = os.path.join(src_dir, 'data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

##########################################

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(3, 196, 3, 1, 1)
        self.layer_norm1 = nn.LayerNorm([32, 32])
        
        self.conv2 = nn.Conv2d(196, 196, 3, 2, 1)
        self.layer_norm2 = nn.LayerNorm([16, 16])

        self.conv3 = nn.Conv2d(196, 196, 3, 1, 1)
        self.layer_norm3 = nn.LayerNorm([16, 16])

        self.conv4 = nn.Conv2d(196, 196, 3, 2, 1)
        self.layer_norm4 = nn.LayerNorm([8, 8])

        self.conv5 = nn.Conv2d(196, 196, 3, 1, 1)
        self.layer_norm5 = nn.LayerNorm([8, 8])

        self.conv6 = nn.Conv2d(196, 196, 3, 1, 1)
        self.layer_norm6 = nn.LayerNorm([8, 8])

        self.conv7 = nn.Conv2d(196, 196, 3, 1, 1)
        self.layer_norm7 = nn.LayerNorm([8, 8])

        self.conv8 = nn.Conv2d(196, 196, 3, 2, 1)
        self.layer_norm8 = nn.LayerNorm([4, 4])

        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

        self.leaky_relu = nn.LeakyReLU()
        self.drop = nn.Dropout2d(p = 0.4)
        self.pool = nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer_norm1(out)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        out = self.layer_norm2(out)
        out = self.leaky_relu(out)

        out = self.conv3(out)
        out = self.layer_norm3(out)
        out = self.leaky_relu(out)

        out = self.conv4(out)
        out = self.layer_norm4(out)
        out = self.leaky_relu(out)

        out = self.conv5(out)
        out = self.layer_norm5(out)
        out = self.leaky_relu(out)

        out = self.conv6(out)
        out = self.layer_norm6(out)
        out = self.leaky_relu(out)

        out = self.conv7(out)
        out = self.layer_norm7(out)
        out = self.leaky_relu(out)

        out = self.conv8(out)
        out = self.layer_norm8(out)
        out = self.leaky_relu(out)

        out = self.pool(out)
        
        out = out.view(out.size(0), -1)

        # determines whether the input is real data or not
        critic_out = self.fc1(out)

        # determines the class of the input
        aux_out = self.fc10(out)

        return critic_out, aux_out

    def layerOutSize(self, inputSize, kernelSize, stride, padding):
        return ((inputSize - kernelSize + 2*padding)/stride) + 1

def load_checkpoint(epoch):
    model_fn = 'epoch_' + str(epoch) + '_cifar10.ckpt'
    model_path = os.path.join(ckpt_dir, model_fn)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

##########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = discriminator()
model.to(device)

if load_epoch != 0:
    load_checkpoint(epoch = load_epoch)

learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##########################################
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)
val_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=8)
##########################################
def train():
    print('\n------------Training------------\n')
    time1 = time.time()
    num_steps = len(train_loader)
    running_loss = 0.0
    correct = 0.
    total = 0.

    for batch_idx, loaded_data in enumerate(train_loader):
        x, y = loaded_data
        x, y = x.to(device), y.to(device)

        # forward pass
        #TODO change when adding critic 
        _, output = model(x)
        # find training accuracy
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

        loss = criterion(output, y)
        
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
        
        # decay learning rate
        if(epoch==50):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate/10.0
        if(epoch==75):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate/100.0

        optimizer.step()
        
        curr_loss = loss.item()
        running_loss += curr_loss
        if (batch_idx+1) % 100 == 0:
            percent_complete = (float(batch_idx+1) / (float(num_steps)))*100.
            print ("Epoch [{}/{}]-----Step [{}/{}]-----Percent Complete: {}-----Current Loss: {:.4f}".format(epoch, num_epochs, batch_idx+1, num_steps, percent_complete, loss.item()))
    train_acc = (correct / total)*100.    
    time2 = time.time()
    epoch_loss = float(running_loss) /  num_steps
    print('Epoch Runtime: {} seconds ----- Epoch Loss: {} ----- Training Accuracy: {}'.format(time2-time1, epoch_loss, train_acc))

    return epoch_loss, train_acc

def validation():
    model.eval()
    num_steps = len(val_loader)
    with torch.no_grad():
        print('\n------------Validation------------\n')
        # find validation accuracy
        correct = 0.
        total = 0.
        for batch_idx, (x,y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            # TODO get both values later
            _, output = model(x)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            if batch_idx % 20 == 0:
                print('Batch Idx: {} / {}'.format(batch_idx, num_steps))
    val_acc = (correct / total) * 100
    print('Validation Accuracy: {}'.format(val_acc))
    return val_acc

def save_checkpoint(final):
    if final:
        ckpt_fn = 'cifar10.model'
    else:
        ckpt_fn = 'epoch_' + str(epoch) + '_cifar10.ckpt'

    ckpt_path = os.path.join(ckpt_dir, ckpt_fn)
    torch.save(model.state_dict(), ckpt_path)

if __name__ == '__main__':
    num_epochs = 100 - load_epoch
    #TODO think of a good way to save this to disk while being able to load arbitrary checkpoints
    # and overwrite the right data
    loss_arr = []
    train_acc_arr = []
    val_acc_arr = []
    epoch_arr = np.linspace(load_epoch, num_epochs, num_epochs)
    # pickup from a checkpoint from epoch 10
    for epoch in range(load_epoch+1, num_epochs):
        epoch_loss, train_acc = train()
        train_acc_arr.append(train_acc)
        loss_arr.append(loss_arr)

        val_acc = validation()
        val_acc_arr.append(val_acc)
        if epoch % 25 == 0:
            save_checkpoint(final = False)

save_checkpoint(final = True)















