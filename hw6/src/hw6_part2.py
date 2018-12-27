import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

##########################################
import sys
import numpy as np
from numpy.linalg import norm
from numpy import shape
import time
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
# set to the value of the epoch checkpoint you want to load, otherwise set to 0
load_epoch = 0

##########################################
if blue_waters:
    src_dir = '/mnt/c/scratch/training/tra392/hw6/src'
    batch_size = 100
    num_workers = 8
else:
    src_dir = 'C:/home/classes/IE534_DL/hw6/src'
    batch_size = 10
    num_workers = 0

ckpt_dir = os.path.join(src_dir, 'checkpoints')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

data_dir = os.path.join(src_dir, 'data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

vis_dir = os.path.join(src_dir, 'visualization')

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

    def forward(self, x, extract_features):
        h = 0
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
        
        if (extract_features==4):
            h = F.max_pool2d(out,8,8)
            h = h.view(-1, 196)

        out = self.conv5(out)
        out = self.layer_norm5(out)
        out = self.leaky_relu(out)

        out = self.conv6(out)
        out = self.layer_norm6(out)
        out = self.leaky_relu(out)

        if (extract_features==6):
            h = F.max_pool2d(out,8,8)
            h = h.view(-1, 196)

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

        return critic_out, aux_out, h

    def layerOutSize(self, inputSize, kernelSize, stride, padding):
        return ((inputSize - kernelSize + 2*padding)/stride) + 1

def load_model():
    #model_fn = 'cifar10.model'
    model_fn = 'tempD260.model'
    model_path = os.path.join(ckpt_dir, model_fn)
    model = torch.load(model_path)
    #model.load_state_dict(state_dict)

##########################################
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root= data_dir, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
testloader = enumerate(testloader)
##########################################

model = discriminator()
load_model()
model.cuda()
model.eval()

##################################

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

def jitter():
    batch_idx, (X_batch, Y_batch) = next(testloader)
    X_batch = Variable(X_batch,requires_grad=True).cuda()
    Y_batch_alternate = (Y_batch + 1)%10
    Y_batch_alternate = Variable(Y_batch_alternate).cuda()
    Y_batch = Variable(Y_batch).cuda()

    ## save real images
    samples = X_batch.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    fig_fn = 'real_images.png'
    fig_path = os.path.join(vis_dir, fig_fn)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

    _, output, blah = model(X_batch, extract_features = 0)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
    print(accuracy)

    ## slightly jitter all input images
    criterion = nn.CrossEntropyLoss(reduce=False)
    loss = criterion(output, Y_batch_alternate)

    gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                            grad_outputs=torch.ones(loss.size()).cuda(),
                            create_graph=True, retain_graph=False, only_inputs=True)[0]

    # save gradient jitter
    gradient_image = gradients.data.cpu().numpy()
    gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
    gradient_image = gradient_image.transpose(0,2,3,1)
    fig = plot(gradient_image[0:100])
    fig_fn = 'gradient_image.png'
    fig_path = os.path.join(vis_dir, fig_fn)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

    # jitter input image
    gradients[gradients>0.0] = 1.0
    gradients[gradients<0.0] = -1.0

    gain = 8.0
    X_batch_modified = X_batch - gain*0.007843137*gradients
    X_batch_modified[X_batch_modified>1.0] = 1.0
    X_batch_modified[X_batch_modified<-1.0] = -1.0

    ## evaluate new fake images
    _, output, blah = model(X_batch_modified, extract_features = 0)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
    print(accuracy)

    ## save fake images
    samples = X_batch_modified.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    fig_fn = 'jittered_images.png'
    fig_path = os.path.join(vis_dir, fig_fn)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

def max_class():   
    batch_idx, (X_batch, Y_batch) = next(testloader)
    X_batch = Variable(X_batch,requires_grad=True)
    X = X_batch.mean(dim=0).cuda()
    X = X.repeat(10,1,1,1)

    Y = torch.arange(10).type(torch.int64)
    Y = Variable(Y).cuda()
    lr = 0.1
   
    weight_decay = 0.001
    for i in range(200):
        _, output, blah = model(X ,extract_features = 0)

        loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()).cuda(),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
        #print(i,accuracy,-loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0

    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples)
    fig_fn = 'max_class.png'
    fig_path = os.path.join(vis_dir, fig_fn)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

def max_features_4():    
    batch_idx, (X_batch, Y_batch) = next(testloader)
    X_batch = Variable(X_batch,requires_grad=True).cuda()
    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size,1,1,1)

    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()

    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        _, blah ,output = model.forward(X, extract_features = 4)

        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()).cuda(),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = (float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-   1.0] = -1.0

    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    fig_fn = 'max_features_4.png'
    fig_path = os.path.join(vis_dir, fig_fn)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

def max_features_6():    
    batch_idx, (X_batch, Y_batch) = next(testloader)
    X_batch = Variable(X_batch,requires_grad=True).cuda()
    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size,1,1,1)

    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()

    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        _, blah ,output = model.forward(X, extract_features = 6)

        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()).cuda(),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = (float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-   1.0] = -1.0

    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    fig_fn = 'max_features_6.png'
    fig_path = os.path.join(vis_dir, fig_fn)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

##################################
if __name__ == '__main__':
    jitter()
    max_class()
    max_features_4()
    max_features_6()