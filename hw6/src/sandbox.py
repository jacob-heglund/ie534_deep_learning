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

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.fc1 = nn.Linear(100, 196*4*4)
        #nn.Conv2d(in, out, kernel, stride, padding)
        self.conv1 = nn.ConvTranspose2d(196, 196, 4, 2, 1)
        self.conv2 = nn.Conv2d(196, 196, 3, 1, 1)
        self.conv3 = nn.Conv2d(196, 196, 3, 1, 1)
        self.conv4 = nn.Conv2d(196, 196, 3, 1, 1)
        self.conv5 = nn.ConvTranspose2d(196, 196, 4, 2, 1)
        self.conv6 = nn.Conv2d(196, 196, 3, 1, 1)
        self.conv7 = nn.ConvTranspose2d(196, 196, 4, 2, 1)
        self.conv8 = nn.Conv2d(196, 3, 3, 1, 1)

        self.batch_norm = nn.BatchNorm2d(196)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(p = 0.4)
        self.pool = nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0)

    def generate(self):
        x = torch.randn([1, 100])
        out = self.fc1(x)
        out = out.reshape(1, 196, 4, 4)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.conv7(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.conv8(out)
        out = self.tanh(out)

        return out

    def layerOutSize(self, inputSize, kernelSize, stride, padding):
        return ((inputSize - kernelSize + 2*padding)/stride) + 1 
# model = generator()
# out = model.generate()

t1 = 2299.130173444748/60 # minutes for 1 epoch
#print(t1, 'minute for 1 epoch with gen_train = 5')
t2 = t1 * 400 / 60
#print(t2, 'hours for total run')
t3 = t2 / 6
print(t3, 'hours for total run with gen_train = 30')









