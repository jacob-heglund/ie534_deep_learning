import os
import numpy as np
from numpy import shape
import time
from random import randint
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from  PIL import Image

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


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 85
learning_rate = 0.0001
#TODO: get this to look like the code i turned in
#TODO: get the training to take around 80 epochs (slow learning rate!)
# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)


class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)

        # use this convolution if its the first one in the set of basic blocks
        # to convert from size 32 of the previous basic block
        self.conv3_1_first = nn.Conv2d(32, 64, 3, 2, 1)
        # if on the first run through a basic block, upsample the number of channels
        # so the feedForward term can be added
        self.upsampleChannels3 = nn.Conv2d(32, 64, 1)
        # otherwise, use this convolution within the set of 4 basic blocks
        self.conv3_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv4_1_first = nn.Conv2d(64, 128, 3, 2, 1)
        self.upsampleChannels4 = nn.Conv2d(64, 128, 1)
        self.conv4_1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv4_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv5_1_first = nn.Conv2d(128, 256, 3, 2, 1)
        self.upsampleChannels5 = nn.Conv2d(128, 256, 1)   
        self.conv5_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, 3, 1, 1)
        
        self.fc = nn.Linear(256, 100)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(4)

    def layer1(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        return out

    def basicBlock2(self, out, first):
        feedForward = out
        out = self.conv2_1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2_2(out)
        out = self.bn2(out)
        feedForward = F.interpolate(feedForward, shape(out)[2])
        out = out + feedForward
        return out
    
    def basicBlock3(self, out, first):
        feedForward = out
        if first:
            feedForward = self.upsampleChannels3(feedForward)
            out = self.conv3_1_first(out)
        else:
            out = self.conv3_1(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        out = self.bn3(out)
        feedForward = F.interpolate(feedForward, shape(out)[2])
        out = out + feedForward
        return out

    def basicBlock4(self, out, first):
        feedForward = out
        if first:
            feedForward = self.upsampleChannels4(feedForward)
            out = self.conv4_1_first(out)
        else:
            out = self.conv4_1(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv4_2(out)
        out = self.bn4(out)
        feedForward = F.interpolate(feedForward, shape(out)[2])
        out = out + feedForward
        return out

    def basicBlock5(self, out, first):
        feedForward = out
        if first:
            feedForward = self.upsampleChannels5(feedForward)
            out = self.conv5_1_first(out)
        else:
            out = self.conv5_1(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv5_2(out)
        out = self.bn5(out)
        feedForward = F.interpolate(feedForward, shape(out)[2])
        out = out + feedForward
        return out

    def forward(self, x):
        out = self.layer1(x)

        out = self.basicBlock2(out, first = True)        
        out = self.basicBlock2(out, first = False)

        out = self.basicBlock3(out, first = True)
        out = self.basicBlock3(out, first = False)
        out = self.basicBlock3(out, first = False)
        out = self.basicBlock3(out, first = False)

        out = self.basicBlock4(out, first = True)
        out = self.basicBlock4(out, first = False)
        out = self.basicBlock4(out, first = False)
        out = self.basicBlock4(out, first = False)

        out = self.basicBlock5(out, first = True)
        out = self.basicBlock5(out, first = False)

        out = self.pool(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        
        return out

    def layerOutSize(self, inputSize, kernelSize, stride, padding):
        return ((inputSize - kernelSize + 2*padding)/stride) + 1

model = MyResNet().to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        # avoid numerical issues
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')





'''
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
'''
