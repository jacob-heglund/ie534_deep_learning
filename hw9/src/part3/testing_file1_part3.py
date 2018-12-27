import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadFrame

import h5py
import cv2

from multiprocessing import Pool
###############################
IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
num_of_epochs = 10

data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)

###############################
model_1 = torch.load('single_frame.model')
model_1.cuda()

model_2 = torch.load('3d_resnet.model')
model_2.cuda()

# save predictions directory
prediction_directory = 'UCF-101-predictions-part3/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory+label+'/'):
        os.makedirs(prediction_directory+label+'/')

acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406],np.float32)
std = np.asarray([0.229, 0.224, 0.225],np.float32)
model_1.eval()
model_2.eval()

###############################
###############################
for i in range(len(test[0])):
    # load data from file
    t1 = time.time()
    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')

    h = h5py.File(filename,'r')
    nFrames = len(h['video'])

    # get closest multiple of 16 below nFrames
    nFrames = int(16.*np.floor(nFrames/16.))
    
    data = np.zeros((nFrames,3,IMAGE_SIZE,IMAGE_SIZE),dtype=np.float32)
    
    # put the frames into an np array called data
    for j in range(nFrames):
        frame = h['video'][j]
        frame = frame.astype(np.float32)
        frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE))
        frame = frame/255.0
        frame = (frame - mean)/std
        frame = frame.transpose(2,0,1)
        data[j,:,:,:] = frame
    h.close()

    ###################
    num_seq = int(nFrames / 16)
    prediction_temp = np.zeros((nFrames, NUM_CLASSES),dtype=np.float32)
    prediction_1 = np.zeros((num_seq, NUM_CLASSES),dtype=np.float32)
    prediction_2 = np.zeros((num_seq, NUM_CLASSES),dtype=np.float32)
    
    loop_i = list(range(0,nFrames,192))
    loop_i.append(nFrames)
    
    count = 0
    count_upper = 0
    for j in range(len(loop_i)-1):
        # each batch consists of 96 frames
        data_batch = data[loop_i[j]:loop_i[j+1]]
     
        # individual frames
        with torch.no_grad():
            x = np.asarray(data_batch,dtype=np.float32)
            x = Variable(torch.FloatTensor(x)).cuda().contiguous()

            output_1 = model_1(x)

        prediction_temp[loop_i[j]:loop_i[j+1]] = output_1.cpu().numpy()

        # sequences
        with torch.no_grad():
            x = np.asarray(data_batch, dtype=np.float32)
            x = Variable(torch.FloatTensor(x).cuda().contiguous())
            x = x.view(-1, 3, 16, 224, 224)
            h = model_2.conv1(x)
            h = model_2.bn1(h)
            h = model_2.relu(h)
            h = model_2.maxpool(h)

            h = model_2.layer1(h)
            h = model_2.layer2(h)
            h = model_2.layer3(h)
            h = model_2.layer4[0](h)
            # h = model.layer4[1](h)

            h = model_2.avgpool(h)

            h = h.view(h.size(0), -1)
            output_2 = model_2.fc(h)
        
        count_upper += output_2.size()[0]

        # number of sequences of 16 frames in the first prediction
        num_pred_1_seq = len(prediction_temp)
        for seq in range(num_pred_1_seq):
            # 16x101
            frames_16 = prediction_temp[seq*16:(seq+1)*16, :]
            # 1x101
            avg_over_16_frames = np.mean(frames_16, axis = 0)
            avg_over_16_frames = avg_over_16_frames.reshape(1, -1)
            prediction_1[count:count+1, :] = avg_over_16_frames 
        
        prediction_2[count:count_upper, :] = output_2.cpu().numpy()        


        pred_final = np.divide(np.add(prediction_1, prediction_2), 2)
        print(np.shape(pred_final))     
        print(pred_final)
        count = count_upper

    ###################
    filename = filename.replace(data_directory+'UCF-101-hdf5/',prediction_directory)
    if(not os.path.isfile(filename)):
        with h5py.File(filename,'w') as h:
            h.create_dataset('predictions',data=prediction)

    # softmax
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j])/np.sum(np.exp(prediction[j]))

    prediction = np.sum(np.log(prediction),axis=0)
    argsort_pred = np.argsort(-prediction)[0:10]

    label = test[1][index]
    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d/%d nFrames:%d t:%f (%f,%f,%f)' 
          % (i, len(test[0]), nFrames,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))
###############################


