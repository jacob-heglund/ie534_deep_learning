'''
Filename: c:\home\classes\IE534_DL\hw4\src\hw4_part2_attempt2.py
Path: c:\home\classes\IE534_DL\hw4\src
Created Date: Tuesday, October 9th 2018, 1:46:34 pm
Author: Jacob Heglund

Copyright (c) 2018 Jacob Heglund

HW4, Part 2: Fine-tune a pre-trained ResNet-18 model and achieve at least 70% test accuracy.
'''
##########################################
import os
import numpy as np
from numpy import shape
import time
import matplotlib.pyplot as plt
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
##########################################
# CIFAR100 dataset info
# image size: 32x32 pixels
# number classes: 100
# x per class: 600
# training x: 50,000
# testing x: 10,000

batchSize = 200
# Image preprocessing modules
trainTransform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validateTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# CIFAR-100 dataset
trainDataset = torchvision.datasets.CIFAR100(root='./data/', train=True, transform=trainTransform, download=True)
#TODO: i may not need the second download (save on disk space a bit)
testDataset = torchvision.datasets.CIFAR100(root='./data/', train=False, transform=validateTransform, download = True)

# Data loader
trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True, num_workers = 0)
testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False, num_workers = 0)
numTrainSteps = len(trainLoader)
############################################

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
        
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 100)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.2)
        self.pool = nn.MaxPool2d(4, 2, 1)

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
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        
        return out

    def layerOutSize(self, inputSize, kernelSize, stride, padding):
        return ((inputSize - kernelSize + 2*padding)/stride) + 1

def plotting(showPlot, savePlot, filename):
    epochArr = np.linspace(1, numEpochs, numEpochs)
    
    ax2 = plt.subplot(212)
    #ax2.set_title('Loss vs. Epochs')
    ax2.plot(epochArr, trainLoss, label = 'Loss - Training Dataset')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)        
    ax2.set_xticks(np.rint(epochArr))

    ax1 = plt.subplot(211, sharex = ax2)
    #ax1.set_title('Accuracy vs. Epochs')
    ax1.plot(epochArr, testAcc, label = 'Accuracy - Validation Dataset')
    ax1.set_ylabel('Accuracy (percent)')
    ax1.set_ylim(ymax = 100, ymin = 0)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    plt.setp(ax1.get_xticklabels(), visible=False)   
    
    if showPlot:    
        plt.show()

    if savePlot:
        plt.savefig(filename, dpi = 200)

def lrSchedule(epoch, decayEpoch, scaling, optimizer, learningRate):    
    if (epoch+1) % decayEpoch == 0:
        learningRate /= 10    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learningRate

##########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters
numEpochs = 100
learningRate = 0.001
# decay learning rate by a factor of 'scaling' every 'decayEpoch' epochs
decayEpoch = 10
scaling = 5
model = MyResNet()
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)


# track accuracy after each epoch
trainLoss = np.zeros([numEpochs, 1])
testAcc = np.zeros([numEpochs, 1])

##########################################
# Train the model
bestModel = {}
bestTestAcc = 0.0
for epoch in range(numEpochs):
    
    print('\n------------Training------------\n')
    model.train()
    time1 = time.time()
    for i, (x, y) in enumerate(trainLoader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        lrSchedule(epoch, decayEpoch, scaling ,optimizer, learningRate)

        # Forward pass
        out = model(x)
        loss = criterion(out, y)
        
        # Backward and optimize
        loss.backward()
        # avoid numerical issues
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        optimizer.step()
        
        if i % 100 == 0:
            print ("Epoch: [{}/{}] ----- Step: [{}/{}] ----- Percent Complete: {:.4f} ----- Loss: {:.4f}"
                   .format(epoch+1, numEpochs, i, numTrainSteps, i / numTrainSteps, loss.item()))
    
    time2 = time.time()
    print('Epoch Runtime: {} seconds'.format(time2-time1))
    trainLoss[epoch] = float(loss)    
    
    print('\n------------Validation------------\n')
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in testLoader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        currTestAcc = 100 * (correct / total)
        print('Validation Accuracy: {} %'.format(currTestAcc))
    testAcc[epoch] = currTestAcc

    if testAcc[epoch] > bestTestAcc:
        bestModel = model.state_dict()
        torch.save(bestModel, 'bestModel_hw4_part1.pt')

    # since blue waters may cut off training early, save loss and accuracy arrays to disk
    np.save('hw4_part1_trainLoss', trainLoss)
    np.save('hw4_part1_testAcc', trainLoss)

# plot loss and accuracy
plotting(showPlot = True, savePlot = True, filename = '4_part2.png')

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
