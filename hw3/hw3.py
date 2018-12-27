"""
HW3: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation. For 10% extra credit, compare dropout test accuracy (i) using the heuristic prediction rule and (ii) Monte Carlo simulation. For full credit, the model should achieve 80-90% Test Accuracy. Submit via Compass (1) the code and (2) a paragraph (in a PDF document) which reports the results and briefly describes the model architecture. Due September 28 at 5:00 PM.

Author: Jacob Heglund
"""
##########################################################
# imports
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
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

##########################################################
class SeqModel(nn.Module):
    def __init__(self):
        super(SeqModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2_dropout = nn.Dropout2d()

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer4_dropout = nn.Dropout2d()

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU())
            
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU())
        self.layer6_dropout = nn.Dropout2d()
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer8_dropout = nn.Dropout2d()
        
        self.layer9 = nn.Sequential(
            nn.Linear(1024, 500),
            nn.ReLU())
        
        self.layer10 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU())

        self.layer11 = nn.Sequential(
            nn.Linear(500, 10),
            nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer2_dropout(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer4_dropout(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer6_dropout(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer8_dropout(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = F.log_softmax(out,0)
        return out
        
##########################################################
class Model(nn.Module):
    # define the network (layers and sizes)
    def __init__(self):
        super(Model, self).__init__()
        # define network layers
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2)
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv3 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv4 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv5 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv8 = nn.Conv2d(64, 64, 3, 1, 0)

        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)
        # BatchNorm2d(out_channels from previous layer)
        self.batchNorm = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.batchNorm(F.relu(self.conv1(x)))
        x = self.drop(self.pool(F.relu(self.conv2(x))))
        x = self.batchNorm(F.relu(self.conv3(x)))
        x = self.drop(self.pool(F.relu(self.conv4(x))))
        x = self.batchNorm(F.relu(self.conv5(x)))
        x = self.drop(F.relu(self.conv6(x)))
        x = self.batchNorm(F.relu(self.conv7(x)))
        x = self.drop(self.batchNorm(F.relu(self.conv7(x))))
        
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x,0)
        return x

if __name__=='__main__':
    ##########################################################
    # define model parameters
    learning_rate = 0.00001

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#    model = Model().to(device)
    model = SeqModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    ##########################################################
    # load data
    batchSize = 80
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    transform_train = transforms.Compose(
        [transforms.RandomVerticalFlip(p = 0.5),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),  
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),  
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])


    trainSet = torchvision.datasets.CIFAR10(root='./data', train = True, download = False, transform = transform_train)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, shuffle = True, num_workers = 0)

    testSet = torchvision.datasets.CIFAR10(root='./data', train = False, download = False, transform = transform_test)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, shuffle = False, num_workers = 0)

    ##########################################################
    # train / test
    numEpochs = 50
    numSteps = 50000/batchSize
    epochVec = np.linspace(0, numEpochs)
    epochAcc = np.zeros(numEpochs)

    for epoch in range(numEpochs):
        time1 = time.time()

        runningLoss = 0.0
        counter = 0
        
        # training
        print('\n------------Training------------\n')
        for i, data in enumerate(trainLoader, 0):
            # set model to have its weights updated
            model.train()

            # import data, send to GPU
            x_CPU, y_CPU = data
            x = Variable(x_CPU.to(device))
            y = Variable(y_CPU.to(device))
            # Forward
            output = model.forward(x)
            loss = criterion(output, y)

            yHat =  torch.argmax(output, dim=1)  
        #Backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runningLoss = loss.item()

            counter += 1
            if counter % 63 == 0:
                print('Epoch: ', epoch+1 ,'----- Percent Complete:', str(float(counter) / 6.3), '----- Running Loss: ', round(runningLoss, 3))

        time2 = time.time()

        print('Epoch runtime: ', time2-time1, ' seconds')
        print('\n------------Training Accuracy------------\n')
        model.eval()
        
        totalCorrect = 0
        total = 50000.0
        for xTrain, yTrain in trainLoader:
            xTrain, yTrain = data
            xTrain = xTrain.to(device)
            yTrain = yTrain.to(device)

            output = model(xTrain)
            _,yHat =  torch.max(output.data,1)

            yHat = np.array(yHat)
            yTrain = np.array(yTrain)
            
            totalCorrect += (yHat == yTrain).sum().item()
        print(totalCorrect)

        acc = (float(totalCorrect)/50000.0)*100.0
        print('Training Accuracy after Epoch ' + str(epoch+1) + ': ' + str(round(acc, 3)) + ' percent')
        

        print('\n------------Testing Accuracy------------\n')
        model.eval()
        with torch.no_grad():
            total = 10000.0            
            totalCorrect = 0
            for xTest, yTest in testLoader:
                xTest = xTest.to(device)
                yTest = yTest.to(device)
                output = model(xTest)
                _,yHat =  torch.max(output.data,1)

                yHat = np.array(yHat)
                yTest = np.array(yTest)
                
                totalCorrect += (yHat == yTest).sum().item()
        print(totalCorrect)

        acc = (float(totalCorrect)/10000.0)*100.0
        print('Test Accuracy after Epoch ' + str(epoch+1) + ': ' + str(round(acc, 3)) + ' percent')




    print('end')



