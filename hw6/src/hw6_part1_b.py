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
    batch_size = 25
    num_workers = 0

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

    def generate(self, x):
        out = self.fc1(x)
        out = out.reshape(batch_size, 196, 4, 4)
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
##########################################
def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

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

def save_checkpoint(final):
    if final:
        ckptG_fn = 'generator.model'
        ckptD_fn = 'discriminator.model'
    else:
        ckptG_fn = 'epoch_' + str(epoch) + '_generator.ckpt'
        ckptD_fn = 'epoch_' + str(epoch) + '_discriminator.ckpt'

    ckptG_path = os.path.join(ckpt_dir, ckptG_fn)
    ckptD_path = os.path.join(ckpt_dir, ckptD_fn)

    torch.save(aG.state_dict(), ckptG_path)
    torch.save(aD.state_dict(), ckptD_path)

def load_checkpoint(epoch):
    model_fn_G = 'epoch_' + str(epoch) + '_generator.ckpt'
    model_fn_D = 'epoch_' + str(epoch) + '_discriminator.ckpt'

    model_path_G = os.path.join(ckpt_dir, model_fn_G)
    model_path_D = os.path.join(ckpt_dir, model_fn_D)

    state_dict_G = torch.load(model_path_G)
    state_dict_D = torch.load(model_path_D)

    aG.load_state_dict(state_dict_G)
    aD.load_state_dict(state_dict_D)

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

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
n_classes = 10
##########################################
aD = discriminator()
aD.cuda()

aG = generator()
aG.cuda()

if load_epoch != 0:
    load_checkpoint(epoch = load_epoch)

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

criterion = nn.CrossEntropyLoss()

np.random.seed(352)
label = np.asarray(list(range(10))*10)
n_z = 100
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()

##########################################
# before epoch training loop starts
loss1 = []
loss2 = []
loss3 = []
loss4 = []
loss5 = []
acc1 = []

gen_train = 1
if __name__ == '__main__':
    num_epochs = 200 - load_epoch
    # Train the model
    for epoch in range(load_epoch + 1, num_epochs):
        print("Epoch: ", epoch)
        start_time = time.time()

        aG.train()
        aD.train()
        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
            # train G
            if((batch_idx%gen_train)==0):
                for p in aD.parameters():
                    p.requires_grad_(False)

                    aG.zero_grad()

                    label = np.random.randint(0,n_classes,batch_size)
                    noise = np.random.normal(0,1,(batch_size,n_z))
                    label_onehot = np.zeros((batch_size,n_classes))
                    label_onehot[np.arange(batch_size), label] = 1
                    noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
                    noise = noise.astype(np.float32)
                    noise = torch.from_numpy(noise)
                    noise = Variable(noise).cuda()
                    fake_label = Variable(torch.from_numpy(label)).cuda()
                    fake_label = fake_label.type(torch.cuda.LongTensor)

                    fake_data = aG.generate(noise)
                    gen_source, gen_class  = aD(fake_data)

                    gen_source = gen_source.mean()
                    
                    gen_class = criterion(gen_class, fake_label)

                    gen_cost = -gen_source + gen_class
                    gen_cost.backward()
                    for group in optimizer_g.param_groups:
                        for p in group['params']:
                            state = optimizer_g.state[p]
                            if('step' in state and state['step']>=1024):
                                state['step'] = 1000

                    optimizer_g.step()


            # train D
            for p in aD.parameters():
                p.requires_grad_(True)

            aD.zero_grad()

            # train discriminator with input from generator
            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()
            fake_label = fake_label.type(torch.cuda.LongTensor)

            with torch.no_grad():
                fake_data = aG.generate(noise)

            disc_fake_source, disc_fake_class = aD(fake_data)

            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, fake_label)

            # train discriminator with input from the discriminator
            real_data = Variable(X_train_batch).cuda()
            real_label = Variable(Y_train_batch).cuda()

            disc_real_source, disc_real_class = aD(real_data)

            prediction = disc_real_class.data.max(1)[1]
            accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, real_label)

            gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)

            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            disc_cost.backward()
            for group in optimizer_d.param_groups:
                for p in group['params']:
                    state = optimizer_d.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000

            optimizer_d.step()

            # within the training loop
            loss1.append(gradient_penalty.item())
            loss2.append(disc_fake_source.item())
            loss3.append(disc_real_source.item())
            loss4.append(disc_real_class.item())
            loss5.append(disc_fake_class.item())
            acc1.append(accuracy)
            if((batch_idx%50)==0):
                print('{} / {}'.format(batch_idx, len(trainloader)))
                print(epoch, batch_idx, "%.2f" % np.mean(loss1), 
                                        "%.2f" % np.mean(loss2), 
                                        "%.2f" % np.mean(loss3), 
                                        "%.2f" % np.mean(loss4), 
                                        "%.2f" % np.mean(loss5), 
                                        "%.2f" % np.mean(acc1))

        # Test the model
        aD.eval()
        print('Testing')
        with torch.no_grad():
            test_accu = []
            for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
                X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

                with torch.no_grad():
                    _, output = aD(X_test_batch)

                prediction = output.data.max(1)[1] # first column has actual prob.
                accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
                test_accu.append(accuracy)
                accuracy_test = np.mean(test_accu)
        print('Accuracy: ',accuracy_test, 'Runtime: ', time.time()-start_time)

        ### save output
        with torch.no_grad():
            aG.eval()
            samples = aG.generate(save_noise)
            samples = samples.data.cpu().numpy()
            samples += 1.0
            samples /= 2.0
            samples = samples.transpose(0,2,3,1)
            aG.train()

        fig = plot(samples)
        plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
        plt.close(fig)

        if(((epoch+1)%1)==0):
            save_checkpoint(final = False)

save_checkpoint(final = True)            
    
