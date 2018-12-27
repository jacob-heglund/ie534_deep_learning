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
import random
from tiny_imagenet_loader_final_results import tinyImageNetDataset

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

#choose the epoch model that you want to get the results from 
epoch_num = 12

val_batch_size = 5
##########################################
# load things from training
#TODO: dataframe misses the first entry in the annotations text file
if blue_waters:
    src_dir = '/mnt/c/scratch/training/tra392/hw5/src'

else:
    src_dir = 'C:/home/classes/IE534_DL/hw5/src'

train_data_dir = 'data/tiny-imagenet-200/train'
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

val_dataset = tinyImageNetDataset(src_dir = src_dir, data_dir = val_data_dir, annotations_path = val_annotations_path, train = False, final_results = True,transform = val_transform)

##########################################
''' final results in words (took about 3 hours to modify the code and get the images looking decent)
- done - sample 5 different images from validation set (each from a different class)
- show top 10 ranked results (i.e. lowest Euclidean distance)
- show the Euclidean distance between query image and ranked images
- show bottom 10 ranked results (i.e. highest Euclidean distance)
- describe at least one way you can improve the network performance (training speed, accuracy, etc.)
-- be super descriptive, there is a possible answer in the paper you can use if you understand it
'''

def final_results(epoch):
    # load data (the if is just so i can fold the code)
    load_data = 1
    if load_data:
        epoch_num = 'epoch_' + str(epoch)
        save_path = os.path.join(backup_dir, epoch_num)

        # load model
        model_fn = 'epoch_' + str(epoch) + '_resnet.ckpt'
        model_path = os.path.join(save_path, model_fn)

        #TODO: change for real dataset
        #model = models.resnet50().to('cuda')
        model = models.resnet18().to('cuda')
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        # load query image classes and paths
        # img_class_data = [class_name (str), img_fp (str)]
        class_buffer_fn = 'epoch_' + str(epoch) + '_class_buffer.txt'
        class_path = os.path.join(save_path, class_buffer_fn)
        with open(class_path, 'r') as f:
            img_class_data = json.load(f)

        #TODO: naming scheme doesn't transfer well between blue waters and my computer...
        classes = []
        paths_old = []
        paths = []
        for i in range(len(img_class_data)):
            classes.append(img_class_data[i][0])
            paths_old.append(img_class_data[i][1])

        src_dir_old = '/mnt/c/scratch/training/tra392/hw5/src'
        src_dir_new = 'C:/home/classes/IE534_DL/hw5/src'

        for i in paths_old:
            x = i.replace(src_dir_old, src_dir_new)
            paths.append(x)
        classes = np.array(classes)
        paths = np.array(paths)
        # load query image maps as tensors
        # q_maps[i, :] = feature maps corresponding to 
        # image in classes[i]
        all_maps_fn = 'epoch_' + str(epoch) + '_all_maps.txt'
        all_maps_path = os.path.join(save_path, all_maps_fn)
        with open(all_maps_path, 'rb') as f:
            q_maps = torch.load(f)
        q_maps = q_maps.detach().numpy().squeeze()

        print('Data for epoch {} loaded'.format(epoch))
    ############################################
    model.eval()
    k = len(q_maps) # number of neighbors
    with torch.no_grad():
        print('\n------------Final Results------------\n')
        # q_maps[i, :] = feature maps corresponding to image in classes[i]
        # classes[i]
        # paths[i]        
        knn_model = KNeighborsClassifier()
        knn_model.fit(q_maps, classes)
        
        # find results
        # with a batch size of 5, run only once to get required images from batch size
        # uses a modified version of the data loader, so that sucks for simplicity lol
        idx = np.arange(0, len(val_dataset))
        random.shuffle(idx)
        rand_idx = idx[0:5]
        figures_near = {}
        figures_far = {}
        
        # get a single validation image
        for i in rand_idx:
            data = val_dataset[i]
            img_array = np.flip(np.transpose(data[0]), 0)
            title = 'img' + str(i)
            figures_near[title] = img_array
            figures_far[title] = img_array

            img_tensor = data[0].unsqueeze(0)
            label = data[1]
            img_tensor = img_tensor.to(device)
            img_map = model(img_tensor)
            img_map_arr = img_map.cpu().numpy().squeeze()

            curr_val_map = img_map_arr.reshape([1,-1])
            dist, idx = knn_model.kneighbors(curr_val_map, n_neighbors=k)
            
            # get the 10 closest and furthers images
            idx_near = idx[:, 0:10]
            idx_far = idx[:, -10:]
            
            dist_near = dist[:, 0:10]
            dist_far = dist[:, -10:]
            
            class_near = classes[idx_near]
            class_far = classes[idx_far]
            
            paths_near = paths[idx_near]
            paths_far = paths[idx_far]

            # add the 10 nearest and furthest images to dicts for plotting later
            for j in range(10):
                img_near = np.flip(np.array(Image.open(paths_near[:, j][0])), 0)
                img_far = np.flip(np.array(Image.open(paths_far[:, j][0])), 0)

                dist_n = dist_near[:, j]
                dist_f = dist_far[:, j]

                figures_near[str(dist_n)] = img_near
                figures_far[str(dist_f)] = img_far

        plot_figures(figures_near, 5, 11)
        plot_figures(figures_far, 5, 11)

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.
    https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title,backgroundcolor = 'white', fontsize = 14.5)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.show()





def plotting(show_plot, save_plot, plot_fn):
    # loss[i] = loss for epoch i
    loss_fn = 'loss.txt'
    loss_path = os.path.join(backup_dir, loss_fn)
    with open(loss_path, 'r') as f:
        loss = json.load(f)    
    epochArr = np.linspace(1, num_epochs, num_epochs)
    
    ax2 = plt.subplot(212)
    #ax2.set_title('Loss vs. Epochs')
    ax2.plot(epochArr, train_loss, label = 'Loss - Training Dataset')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)        
    ax2.set_xticks(np.rint(epochArr))

    ax1 = plt.subplot(211, sharex = ax2)
    #ax1.set_title('Accuracy vs. Epochs')
    ax1.plot(epochArr, test_acc, label = 'Accuracy - Validation Dataset')
    ax1.set_ylabel('Accuracy (percent)')
    ax1.set_ylim(ymax = 100, ymin = 0)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    plt.setp(ax1.get_xticklabels(), visible=False)   
    
    if show_plot:    
        plt.show()

    if save_plot:
        plt.savefig(plot_fn, dpi = 200)

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
    final_results(epoch_num)




