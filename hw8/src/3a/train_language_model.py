import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import time
import os
import sys
import io

from RNN_language_model import RNN_language_model
##########################################
# program controls
blue_waters = 0
##########################################
# set important directories
if blue_waters:
    src_dir = '/mnt/c/scratch/training/tra392/hw8/src'
    train_directory = '/projects/training/bauh/NLP/aclImdb/train/'
    test_directory = '/projects/training/bauh/NLP/aclImdb/test/'

else:
    src_dir = 'C:/home/classes/IE534_DL/hw8/src'

##########################################
vocab_size = 8000

x_train = []
with io.open('./preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('./preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1
print('Data Loaded')
##########################################
model = RNN_language_model(vocab_size = vocab_size, no_of_hidden_units = 500)
model.cuda()

# opt = 'sgd'
# LR = 0.01
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

batch_size = 50
no_of_epochs = 20
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []

##########################################
for epoch in range(0,75):
    print('-----Training-----')
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR/10.0

    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = 50
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl<sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = Variable(torch.LongTensor(x_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(x_input, train = True)

        loss.backward()

        norm = nn.utils.clip_grad_norm_(model.parameters(),2.0)

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        optimizer.step()   # update gradients
        
        values,prediction = torch.max(pred,1)
        prediction = prediction.cpu().data.numpy()
        accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
        epoch_acc += accuracy
        epoch_loss += loss.data.item()
        epoch_counter += batch_size
        
        if (i+batch_size) % 1000 == 0 and epoch==0:
           print('Step: ', str(i+batch_size) + '/' + str(L_Y_train), '---- Accuracy: ', accuracy/batch_size, '---- Time: ', float(time.time()-time1))
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print('Epoch: ', epoch, '---- Accuracy: ', (epoch_acc*100.0), '---- Loss: ', epoch_loss, '---- Runtime: ', float(time.time()-time1))

    ## test
    print('-----Validation-----')
    if((epoch+1)%1==0):
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
        
        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):
            sequence_length = 100
            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl<sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            x_input = Variable(torch.LongTensor(x_input)).cuda()

            with torch.no_grad():
                pred = model(x_input,train=False)
            
            values,prediction = torch.max(pred,1)
            prediction = prediction.cpu().data.numpy()
            accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
            epoch_acc += accuracy
            epoch_loss += loss.data.item()
            epoch_counter += batch_size
            #train_accu.append(accuracy)
            if (i+batch_size) % 1000 == 0 and epoch==0:
                print('Step: ', str(i+batch_size) + '/' + str(L_Y_test), '---- Accuracy: ', accuracy/batch_size)
        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print('Accuracy: ', (epoch_acc*100.0), '---- Loss: ', epoch_loss, '---- Validation Runtime: ', time_elapsed)
    torch.cuda.empty_cache()

    if(((epoch+1)%2)==0):
        torch.save(model,'temp.model')
        torch.save(optimizer,'temp.state')
        data = [train_loss,train_accu,test_accu]
        data = np.asarray(data)
        np.save('data.npy',data)
torch.save(model,'language.model')
