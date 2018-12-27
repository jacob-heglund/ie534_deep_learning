'''
loz
Filename: c:\home\classes\IE534_DL\hw4\src\hw4_part2_attempt2.py
Path: c:\home\classes\IE534_DL\hw4\src
Created Date: Tuesday, October 9th 2018, 1:46:34 pm
Author: Jacob Heglund

Copyright (c) 2018 Jacob Heglund

HW4, Part 2: Fine-tune a pre-trained ResNet-18 model and achieve at least 70% test accuracy.
'''
#TODO: use proper weight init
 

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

batchSize = 100
# Image preprocessing modules
trainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validateTransform = transforms.Compose([
    transforms.Resize(224),
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
numEpochs = 30
learningRate = 0.001
# decay learning rate by a factor of 'scaling' every 'decayEpoch' epochs
decayEpoch = 8
scaling = 10
def resnet18(pretrained = True) :
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir = './'))
    return model

model = models.resnet18(pretrained = True)
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
        torch.save(bestModel, 'bestModel.pt')


# plot loss and accuracy
plotting(showPlot = True, savePlot = True, filename = '4_part2.png')

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')

