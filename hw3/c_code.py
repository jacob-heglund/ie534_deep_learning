import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd

#data tranformation and augmentation
transform_train = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                     (0.24703223, 0.24348513, 0.26158784))
                                ])

transform_test = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                     (0.24703223, 0.24348513, 0.26158784))
                                ])

train_dataset = torchvision.datasets.CIFAR10(root='~/ie534',
                                             train=True,
                                             transform=transform_train,
                                             download=False)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root='~/ie534',
                                             train=False,
                                             transform=transform_test,
                                             download=False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=50,shuffle=False)
##############################################################################################
class CNN(nn.Module):
    def __init__(self):
        #Part1
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, padding=2)
        #Part2
        self.conv3 = nn.Conv2d(64, 128, 4, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 4, padding=2)
        #Part3
        self.conv5 = nn.Conv2d(128, 128, 4, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, 3, padding=0)
        #Part4
        self.conv7 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        #Part5
        self.ln1 = nn.Linear(1024, 500)
        self.ln2 = nn.Linear(500,500)
        self.ln3 = nn.Linear(500, 10)
    def forward(self, x):
        #Part1
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = F.dropout(F.max_pool2d(out, 2),0.4)
        #Part2
        out = F.relu(self.bn2(self.conv3(out)))
        out = F.relu(self.conv4(out))
        out = F.dropout(F.max_pool2d(out, 2),0.4)
        #Part3
        out = F.relu(self.bn3(self.conv5(out)))
        out = F.relu(self.conv6(out))
        out = F.dropout(out, 0.4)
        #Part4
        out = F.relu(self.bn4(self.conv7(out)))
        out = F.relu(self.bn5(self.conv8(out)))
        #Part5
        out = out.view(out.size(0), -1)
        out = F.relu(self.ln1(out))
        out = F.relu(self.ln2(out))
        out = self.ln3(out)
        return(out)

cnn = CNN()
use_cuda = torch.cuda.is_available()

if use_cuda:
    cnn.cuda()
    cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
train_accu = []
test_accu = []


def train(epoch):
    print("Training: Current Epoch:", epoch)
    #training
    cnn.train()
    correct_save = 0
    total_num = 0
    current_accu = 0
    
    for index, (xx, yy) in enumerate(trainloader):
        if use_cuda:
            xx, yy = xx.cuda(), yy.cuda()
        xx, yy = Variable(xx), Variable(yy)
        optimizer.zero_grad()
        after = cnn(xx)
        loss = loss_func(after, yy)
        loss.backward()
        if(epoch>6):
           for group in optimizer.param_groups:
              for p in group['params']:
                  state = optimizer.state[p]
                  if(state['step']>1024):
                     state['step']=1000
        optimizer.step()
        predict = after.data.max(1)[1]
        total_num += yy.size(0)
        correct_save += predict.eq(yy.data).sum()
        current_accu = float(100.*correct_save)/float(total_num)
    print("Training Accuray:", current_accu )
    train_accu.append(current_accu)


def test(epoch):
    print('Testing: Current Epoch:', epoch)
    #testing
    cnn.eval()
    correct_save = 0
    total_num = 0
    current_accu = 0
    #let pytorch not update gradient
    for index, (xx, yy) in enumerate(testloader):
        if use_cuda:
            xx, yy = xx.cuda(), yy.cuda()
        xx, yy = Variable(xx), Variable(yy)
        after = cnn(xx)
        predict = after.data.max(1)[1]
        total_num += yy.size(0)
        correct_save += predict.eq(yy.data).sum()
        current_accu = float(100.*correct_save)/float(total_num)
        #print("Testinging Step:", index, " Testing Accuracy:", current_accu)
    print("Testing Accuray:", current_accu )
    test_accu.append(current_accu)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters())
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(20):
    train(epoch)
    test(epoch)

a = np.array(train_accu)
b = np.array(test_accu)

df = pd.DataFrame({"train_accu" : a, "test_accu" : b})
df.to_csv("result.csv", index=False)