"""
HW4, Part 1 - ResNets
Build the Residual Network specified in Figure 1 and achieve at least 60% test accuracy.
In the homework, you should define your “Basic Block” as shown in Figure 2. For each
weight layer, it should contain 3 × 3 filters for a specific number of input channels and
output channels. The output of a sequence of ResNet basic blocks goes through a max
pooling layer with your own choice of filter size, and then goes to a fully-connected
layer. The hyperparameter specification for each component is given in Figure 1. Note
that the notation follows the notation in He et al. (2015).

Author: Jacob Heglund
"""

#TODO:
# add saving / loading capability
##############################################################################################
# regular python stuff
import os
import numpy as np
from numpy import shape
import time
from random import randint
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from  PIL import Image

# torch neural net stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
import torchvision.transforms as transforms
##############################################################################################
# CIFAR100 dataset info
# image size: 32x32 pixels
# number classes: 100
# images per class: 600
# training images: 50,000
# testing images: 10,000
batchSize = 100

transformTrain = transforms.Compose([
                                transforms.RandomHorizontalFlip(p = 0.3),
                                transforms.RandomVerticalFlip(p = 0.3),
                                transforms.RandomCrop(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
                                ])

transformTest = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
                                ])

trainSet = torchvision.datasets.CIFAR100(root='./data', train = True, download = True, transform = transformTrain)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, shuffle = True, num_workers = 0)
trainSetSize = len(trainSet)

testSet = torchvision.datasets.CIFAR100(root='./data', train = False, download = True, transform = transformTest)
testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, shuffle = False, num_workers = 0)
testSetSize = len(testSet)

##############################################################################################
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


def printTrainData(counter, numTrainingSteps, epoch, trainingLoss):
    if (counter % (numTrainingSteps / 10)) == 0:
        print('Epoch: ', epoch+1 , \
        '----- Percent Complete:', ((counter/numTrainingSteps)*100), \
        '----- Training Loss: ', round(trainingLoss, 3))

def train(epoch):
    print('\n------------Training------------\n')
    model.train()
    correct = 0.
    total = 0.
    currentTrainAcc = 0.
    counter = 0

    time1 = time.time()
    for x, y in trainLoader:
        x, y = Variable(x.to(device)), Variable(y.to(device))
        
        optimizer.zero_grad()
        out = model.forward(x)
        loss = lossFunction(out, y)
        # calculate gradients
        loss.backward()
        # update gradients
        optimizer.step()
        pred = out.data.max(1)[1]
        trainingLoss = loss.item()
        
        yHat = np.array(pred)
        yHat = np.reshape(yHat, batchSize)
        yArr = np.array(y)

        total += shape(y)[0]
        correct += (yHat == yArr).sum().item()

        currentTrainAcc = (correct / total) * 100.
        counter += 1
        printTrainData(counter, numTrainingSteps, epoch, trainingLoss)

    time2 = time.time()
    print('Training Accuracy after Epoch ' + str(epoch+1) + ': ' + str(round(currentTrainAcc, 3)) + ' percent')
    print('Epoch runtime: ', time2-time1, ' seconds')
    trainAcc[epoch] = currentTrainAcc
    trainLoss[epoch] = trainingLoss

    print('\n------------Validation------------\n')
    model.eval()
    correct = 0.
    total = 0.
    currentTestAcc = 0.
    with torch.no_grad():
        for x, y in testLoader:
            #TODO: stuff I've done to try fixing this:
            # changed Variable(x) to x
            # put correct, total, and currentTestAcc definitions before torch.no_grad()
            x, y = x.to(device), y.to(device)
            out = model.forward(x)
            loss = lossFunction(out, y)
            _,pred = torch.max(out.data,1)
            testingLoss = loss.item()
            total += shape(y)[0]
            
            # yHat = np.array(pred)
            # yHat = np.reshape(yHat, batchSize)
            # yArr = np.array(y)            
            # correct += (yHat == yArr).sum().item()
            correct += (pred==y).sum().item()
        currentTestAcc = (correct / total)*100.
        
    print('Validation Accuracy after Epoch ' + str(epoch+1) + ': ' + str(round(currentTestAcc, 3)) + ' percent')
    testAcc[epoch] = currentTestAcc
    testLoss[epoch] = testingLoss

    
def test(epoch):
    print('\n------------Validation------------\n')
    model.eval()
    correct = 0.
    total = 0.
    currentTestAcc = 0.
    with torch.no_grad():
        for x, y in testLoader:
            #TODO: stuff I've done to try fixing this:
            # changed Variable(x) to x
            # put correct, total, and currentTestAcc definitions before torch.no_grad()
            x, y = x.to(device), y.to(device)
            out = model.forward(x)
            loss = lossFunction(out, y)
            _,pred = torch.max(out.data,1)
            testingLoss = loss.item()
            total += shape(y)[0]
            
            # yHat = np.array(pred)
            # yHat = np.reshape(yHat, batchSize)
            # yArr = np.array(y)            
            # correct += (yHat == yArr).sum().item()
            correct += (pred==y).sum().item()
        currentTestAcc = (correct / total)*100.
        
    print('Testing Accuracy after Epoch ' + str(epoch+1) + ': ' + str(round(currentTestAcc, 3)) + ' percent')
    testAcc[epoch] = currentTestAcc
    testLoss[epoch] = testingLoss

def layerOutSize(inputSize, kernelSize, stride, padding):
    # get the output size of a convolutional layer
    return ((inputSize - kernelSize + 2*padding)/stride) + 1

def plotting(showPlot, savePlot, filename):
    if showPlot:
        ax2 = plt.subplot(212)
        #ax2.set_title('Loss vs. Epochs')
        ax2.plot(epochCount, testLoss, label = 'Loss - Validation Dataset')
        ax2.plot(epochCount, trainLoss, label = 'Loss - Training Dataset')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels)        
        ax2.set_xticks(np.rint(epochCount))

        ax1 = plt.subplot(211, sharex = ax2)
        #ax1.set_title('Accuracy vs. Epochs')
        ax1.plot(epochCount, testAcc, label = 'Accuracy - Validation Dataset')
        ax1.plot(epochCount, trainAcc, label = 'Accuracy - Training Dataset')
        ax1.set_ylabel('Accuracy (percent)')
        ax1.set_ylim(ymax = 100, ymin = 0)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        plt.setp(ax1.get_xticklabels(), visible=False)   
        plt.show()

        if savePlot:
            plt.savefig(filename, dpi = 200)

##############################################################################################
# define model parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MyResNet().to(device)
learningRate = .001
lossFunction = nn.CrossEntropyLoss()
#TODO: try using SGD
optimizer = optim.Adam(model.parameters(),lr=learningRate)
#TODO: use learning rate scheduler
learningRate = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = .1)

numEpochs = 10
# number of steps taken within each epoch 
numTrainingSteps = float(trainSetSize) / float(batchSize)

# track accuracy after each epoch
trainAcc = np.zeros([numEpochs, 1])
trainLoss = np.zeros([numEpochs, 1])
testAcc = np.zeros([numEpochs, 1])
testLoss = np.zeros([numEpochs, 1])

epochCount = np.linspace(1, numEpochs, numEpochs)

##############################################################################################
for epoch in range(numEpochs):
    train(epoch)
    #test(epoch)

plotting(showPlot = True, savePlot = True, filename = 'training.png')

print('end')




