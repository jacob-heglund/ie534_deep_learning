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

num_epochs = 13

# set these high since we're not doing backprop
train_batch_size = 10
val_batch_size = 10
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
# validation (in words)
'''
it took a full day of work (8 hours+) to develop a way to get the accuracy in a reasonable amount of time 

- done-grab an image from the validation set (modify dataloader to do this)
- done-using val_annotations (import as pd dataframe), we have the classes for each image in val set
- done-send the image through the network to get a feature map
- done-get the 30 'closest' images (smallest L2 norm) to the validation image
- done-get the original classes of each of those 30 images
- done-have a counter for total number of images seen, and tota number of correct images
- done-accuracy = (correct / total)*100
- also do this with images in the training set to get a 'training accuracy'
'''
'''
# uses the norm method -> waaaaaay to0 slow... (takes like 2 hours for a single epoch...)
def validation(q_maps):
    model.eval()
    correct = 0.
    total = 0.
    k = 30 # number of "closest" images to validation image
    with torch.no_grad():
        print('\n------------Validation------------\n')
        
        q_maps = np.transpose(q_maps)
        # q_maps[i, :] = feature maps corresponding to image in classes[i]
        # classes[i]
        # paths[i]
        for step, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)    
            img_maps = model(imgs)
            img_map_arr = img_maps.cpu().numpy().squeeze()
            time1 = time.time()
            # get a single validation image map and find the 30 nearest results
            # in the database

            for i in range(val_batch_size):
                class_curr = labels[i]
                class_curr = np.repeat(class_curr, k)
                
                map_curr = img_map_arr[i, :]
                map_curr = np.reshape(map_curr, [-1,1])
                val_map_curr = np.tile(map_curr, shape(q_maps)[1])
                
                # calculate norm between a single feature embedding and 
                # all of the training embeddings
                map_norm = norm(q_maps - val_map_curr, axis = 0)
                
                # get indices of 30 smallest values from map_norm
                idx = np.argpartition(map_norm, k)
                idx_k = idx[:k]
                
                dist_k = map_norm[idx_k]
                classes_k = classes[idx_k]
                
                correct += np.count_nonzero(class_curr == classes_k)
                total += k
                
                print(i)
            print('Step {} / {}'.format(step, len(val_loader)))

        accuracy = (correct / total)
        return(accuracy)
'''

def validation(epoch):
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

        classes = []
        paths = []
        for i in range(len(img_class_data)):
            classes.append(img_class_data[i][0])
            paths.append(img_class_data[i][1])

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
    k = 30 # number of neighbors
    
    with torch.no_grad():
        print('\n------------Validation------------\n')
        # q_maps[i, :] = feature maps corresponding to image in classes[i]
        # classes[i]
        # paths[i]        
        knn_model = KNeighborsClassifier()
        knn_model.fit(q_maps, classes)
        ''' this wasn't working at  3 PM the day the homework was due, so I abandoned it 
        # find training accuracy for an epoch
        calc_train_acc = 1
        if calc_train_acc:
            correct = 0.
            total = 0.
            time1 = time.time()
            for step, (imgs, labels) in enumerate(train_loader):
                print('training')
                imgs = imgs.to(device)    
                img_maps = model(imgs)
                img_map_arr = img_maps.cpu().numpy().squeeze()
                
                # get a single validation image map and find the 30 nearest results
                # in the database
                for j in range(train_batch_size):
                    curr_label = labels[j]
                    curr_class = np.repeat(curr_label, k)
                    
                    curr_val_map = img_map_arr[j, :].reshape([1,-1])
                    dist, idx = knn_model.kneighbors(curr_val_map, n_neighbors=k)
                    classes_k = classes[idx]
                    
                    correct += np.count_nonzero(curr_class == classes_k)
                    total += k
                accuracy_running = (correct / total)*100
                print('Step: {} / {} ----- Accuracy: {}'.format(step, len(val_loader), accuracy_running))

            train_acc = (correct / total) * 100
            time2 = time.time
            print('Runtime: {} minutes----- Accuracy: {}'.format((time2-time1)/60., train_acc))

        '''
        # find validation accuracy
        calc_val_acc = 1
        if calc_val_acc:
            correct = 0.
            total = 0.
            time1 = time.time()
            for step, (imgs, labels) in enumerate(val_loader):
                imgs = imgs.to(device)    
                img_maps = model(imgs)
                img_map_arr = img_maps.cpu().numpy().squeeze()
                
                # get a single validation image map and find the 30 nearest results
                # in the database
                for j in range(val_batch_size):
                    curr_label = labels[j]
                    curr_class = np.repeat(curr_label, k)
                    
                    curr_val_map = img_map_arr[j, :].reshape([1,-1])
                    dist, idx = knn_model.kneighbors(curr_val_map, n_neighbors=k)
                    classes_k = classes[idx]
                    
                    correct += np.count_nonzero(curr_class == classes_k)
                    total += k
                accuracy_running = (correct / total)*100
                print('Step: {} / {} ----- Accuracy: {}'.format(step, len(val_loader), accuracy_running))

            val_acc = (correct / total) * 100
            time2 = time.time
            print('Runtime: {} minutes----- Accuracy: {}'.format((time2-time1)/60., val_acc))
       
        return train_acc, val_acc

def plotting(show_plot, save_plot, acc_fn, loss_fn):
    ax2 = plt.subplot(212)
    #ax2.set_title('Loss vs. Epochs')
    ax2.plot(epoch_arr, train_acc, label = 'Accuracy - Training Dataset')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (percent)')
    ax2.set_ylim(ymax = 100, ymin = 0)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)        
    ax2.set_xticks(np.rint(epoch_arr))

    ax1 = plt.subplot(211, sharex = ax2)
    #ax1.set_title('Accuracy vs. Epochs')
    ax1.plot(epoch_arr, val_acc, label = 'Accuracy - Validation Dataset')
    ax1.set_ylabel('Accuracy (percent)')
    ax1.set_ylim(ymax = 100, ymin = 0)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    plt.setp(ax1.get_xticklabels(), visible=False)   
    
    if show_plot:    
        plt.show()

    if save_plot:
        plt.savefig(acc_fn, dpi = 200)

    # loss[i] = loss for epoch i
    loss_fn = 'loss.txt'
    loss_path = os.path.join(backup_dir, loss_fn)
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
    #epoch_arr = np.arange(0, num_epochs, 1)
    
    #for epoch in range(num_epochs):
    epoch = 11
    print('Epoch:', epoch)
    train_acc, val_acc = validation(epoch)

    plotting(show_plot = True, save_plot = True, acc_fn = 'accuracy.png', loss_fn = 'loss.png')


