##########################################
import os
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
from tiny_imagenet_loader import tinyImageNetDataset
from sklearn.neighbors import KNeighborsClassifier

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
# controls for the program
# 0 for not blue waters, 1 for blue waters
blue_waters = 0

num_epochs = 14

# set these high since we're not doing backprop
train_batch_size = 200
val_batch_size = 100
##########################################
# load things from training
if blue_waters:
    src_dir = '/mnt/c/scratch/training/tra392/hw5/src/'

else:
    src_dir = 'C:/home/classes/IE534_DL/hw5/src'

train_data_dir = 'data/tiny-imagenet-200/train/'
val_data_dir = 'data/tiny-imagenet-200/val'
train_annotations_path = os.path.join(src_dir, 'img_df.csv')
val_annotations_path = os.path.join(src_dir, 'data/tiny-imagenet-200/val/val_annotations.txt')
backup_dir = os.path.join(src_dir, 'backup_resnet18')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda':
    print('cuda -> Using GPU powerrrrrrr')
else:
    print('lame, cpu...')

val_transform =transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])

train_dataset = tinyImageNetDataset(src_dir = src_dir, data_dir = train_data_dir, annotations_path = train_annotations_path, train = True,transform = val_transform)

val_dataset = tinyImageNetDataset(src_dir = src_dir, data_dir = val_data_dir, annotations_path = val_annotations_path, train = False, transform = val_transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle=True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = val_batch_size, shuffle = True, num_workers = 0)

##########################################
def plotting(show_plot, save_plot, acc_fn, loss_fn):
        # loss[i] = loss for epoch i
    loss_load_fn = 'loss.txt'
    loss_path = os.path.join(backup_dir, loss_load_fn)
    with open(loss_path, 'r') as f:
        loss = json.load(f)    
    loss = np.array(loss)
    
    ax3 = plt.subplot(111)
    #ax1.set_title('Accuracy vs. Epochs')
    ax3.plot(epoch_arr, loss, label = 'Loss - Training Dataset')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels)

    if show_plot:    
        plt.show()

    if save_plot:
        plt.savefig(loss_fn, dpi = 200)


def showImage(q_gpu, p_gpu, n_gpu):
    q_cpu = q_gpu.cpu()
    p_cpu = p_gpu.cpu()
    n_cpu = n_gpu.cpu()

    q_arr = q_cpu.numpy().transpose().squeeze()
    p_arr = p_cpu.numpy().transpose().squeeze()
    n_arr = n_cpu.numpy().transpose().squeeze()
    
    img_arr = np.concatenate((q_arr, p_arr, n_arr), axis = 1)
    plt.imshow(img_arr)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    epoch_arr = np.arange(0, num_epochs, 1)
    plotting(show_plot = True, save_plot = True, acc_fn = 'accuracy.png', loss_fn = 'loss.png')


