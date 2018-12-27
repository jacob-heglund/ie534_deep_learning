"""
HW4, part 2: Fine-tune a pre-trained ResNet-18 model and achieve at least 70% test accuracy.

Author: Jacob Heglund
"""
#TODO: i get testing accuracy higher than training accuracy, which seems really wrong...
# run this on Blue Waters for like 50 epochs
###################################################
# regular python stuff
import os
import numpy as np
from numpy import shape
import time
import matplotlib.pyplot as plt
from  PIL import Image
from copy import deepcopy

# torch neural net stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
import torchvision.transforms as transforms

###################################################
# CIFAR100 dataset info
# image size: 32x32 pixels
# number classes: 100
# images per class: 600
# training images: 50,000
# testing images: 10,000
batchSize = 25

transformTrain = transforms.Compose([
    # crops, then upscales to the proper size
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformTest = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


trainSet = torchvision.datasets.CIFAR100(root='./data', train = True, download = True, transform = transformTrain)
trainSetSize = len(trainSet)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, shuffle = True, num_workers = 0)

testSet = torchvision.datasets.CIFAR100(root='./data', train = False, download = True, transform = transformTest)
testSetSize = len(testSet)
testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, shuffle = True, num_workers = 0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###################################################
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
            #TODO: shape of out is 50 x 1000
            loss = lossFunction(out, y)
            pred = out.data.max(1)[1]
            testingLoss = loss.item()
            total += shape(y)[0]
            
            yHat = np.array(pred)
            yHat = np.reshape(yHat, batchSize)
            yArr = np.array(y)            
            correct += (yHat == yArr).sum().item()
        
        currentTestAcc = (correct / total)*100.
        
    print('Testing Accuracy after Epoch ' + str(epoch+1) + ': ' + str(round(currentTestAcc, 3)) + ' percent')
    testAcc[epoch] = currentTestAcc
    testLoss[epoch] = testingLoss

def printTrainData(counter, numTrainingSteps, epoch, trainingLoss):
    if (counter % (numTrainingSteps / 10)) == 0:
        print('Epoch: ', epoch+1 , \
        '----- Percent Complete:', ((counter/numTrainingSteps)*100), \
        '----- Training Loss: ', round(trainingLoss, 3))

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

'''
def train_model(model, criterion, optimizer, scheduler, numEpochs):

    bestModelWeights = deepcopy(model.state_dict())
    bestAcc = 0.0
    steps = trainDatasetSize / batchSize

    for epoch in range(numEpochs):
        print('Epoch', str(epoch), '/', str(numEpochs), '\n')
        epochStartTime = time.time()

        print('\n------------Training------------\n')
        runningLoss = 0.0
        correct = 0
        counter = 0

        scheduler.step()
        model.train()
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            runningLoss += loss.item() * inputs.size(0)
            correct += torch.sum(predictions == labels.data)
            counter += 1
            print(counter)
            if (counter % steps/10) == 0:
                print('{} percent complete'.format((counter / steps)*100))
        
        # check accuracy and loss
        epochLoss = runningLoss / trainDatasetSize
        epochAcc = correct.float() / trainDatasetSize
        epochEndTime = time.time()
        print('Loss: {:.4f} ----- Accuracy: {:.4f}'.format(epochLoss, epochAcc))
        print('Epoch Runtime: {:.4f}'.format(epochEndTime - epochStartTime))
        #######################################################################
        print('\n------------Evaluating------------\n')
        runningLoss = 0.0
        correct = 0

        model.eval()
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                #TODO: get rid of all gradient stuff here since we're evaluating
            runningLoss += loss.item() * inputs.size(0)
            correct += torch.sum(predictions == labels.data)

        # check accuracy and loss
        epochLoss = runningLoss / trainDatasetSize
        epochAcc = correct.float() / trainDatasetSize
        print('Loss: {:.4f} ----- Accuracy: {:.4f}'.format(epochLoss, epochAcc))
        
        if epochAcc > bestAcc:
            bestAcc = epochAcc
            bestModelWeights = deepcopy(model.state_dict())

    print('Best Accuracy: {:4f}'.format(bestAcc))
    model.load_state_dict(bestModelWeights)
    return model
'''
###################################################
# define model parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained = True)
model = model.to(device)

lossFunction = nn.CrossEntropyLoss()
#TODO: try using SGD
optimizer = optim.Adam(model.parameters())
#TODO: use learning rate scheduler
learningRate = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = .1)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 10 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

numEpochs = 15
# number of steps taken within each epoch 
numTrainingSteps = float(trainSetSize) / float(batchSize)

# track accuracy after each epoch
trainAcc = np.zeros([numEpochs, 1])
trainLoss = np.zeros([numEpochs, 1])
testAcc = np.zeros([numEpochs, 1])
testLoss = np.zeros([numEpochs, 1])

epochCount = np.linspace(1, numEpochs, numEpochs)
###################################################
#model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, numEpochs=25)
for epoch in range(numEpochs):
    train(epoch)
    test(epoch)

plotting(showPlot = True, savePlot = True, filename = 'training.png')

print('end')

















