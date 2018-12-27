'''
Use this to generate image triplets for similarity ranking.  Triplets consist of a query image,
positive image (of the same class as query image), and negative image (different class than query image)
'''
import os
import numpy as np
from random import randint
import pandas as pd
import time

# generate two directories randomly
# generate training and validation sets too!
sourceDirTrain = 'c:\home\classes\IE534_DL/hw5/src/data/tiny-imagenet-200/train'
numClasses = 200
numImagesPerClass = 500

def childDirList(rootDir):
    ''' return a list of the immediate child directories of rootDir''' 
    dirList = []
    for subdir in next(os.walk(rootDir))[1]:
        dirList.append(subdir)
    return dirList

def fileList(rootDir):
    '''generate a list of all filenames within a directory'''
    fileList = []
    for files in next(os.walk(rootDir)):
        if files != []:
            fileList.append(files)
    return fileList[1]

def generateTrainTriplets(sourceDirTrain):
    '''
    generates image triplets and copies them to a new folder
    returns: 
        tripletPath = [queryImgPath, posImgPath, negImgPath]
        tripletFn = [queryImgfn, posImgfn, negImgfn]
    '''
    # get a list of all immediate child directories
    trainDirs = childDirList(sourceDirTrain)
    triplet_idx = 0

    numClasses = 200
    numImagesPerClass = 500
    for i in trainDirs:
        pos_dir = i
        pos_class_path = os.path.join('c:\home\classes\IE534_DL/hw5/src/data/tiny-imagenet-200/train', pos_dir, 'images')

        # do query and positive images
        images = fileList(pos_class_path)
        for j in images:
            query_img_fn = j
            query_img_path = os.path.join(pos_class_path, query_img_fn)
            
            # generate randomly sampled positive example (different image in same class)
            sample_pos_img = True
            while sample_pos_img:
                pos_rand_img_fn = images[randint(0, numImagesPerClass-1)]
                
                if pos_rand_img_fn != query_img_fn:
                    pos_img_fn = pos_rand_img_fn
                    pos_img_path = os.path.join(pos_class_path, pos_img_fn)
                    sample_pos_img = False

            # generate randomly sampled negative class and image
            sample_neg_dir = True
            while sample_neg_dir:
                neg_dir_idx = randint(0, numClasses-1)
            
                if trainDirs[neg_dir_idx] != pos_dir:
                    neg_dir = trainDirs[neg_dir_idx]
                    sample_neg_dir = False
                    
                    neg_class_path = os.path.join('c:\home\classes\IE534_DL/hw5/src/data/tiny-imagenet-200/train', neg_dir, 'images')
                    neg_images = fileList(neg_class_path)
                    neg_img_fn = neg_images[randint(0, numImagesPerClass-1)] 
                    neg_img_path = os.path.join(neg_class_path, neg_img_fn)

                    triplet_fn = [query_img_fn, pos_img_fn, neg_img_fn]
                    triplet_path = [query_img_path, pos_img_path, neg_img_path]

def generateImgIdx():
    '''
    generates a unique index for each image and stores it with the fn in a csv
    '''
    root_dir = 'c:\home\classes\IE534_DL/hw5/src/data/tiny-imagenet-200/train'
    
    # get a list of all immediate child directories
    train_dirs = childDirList(root_dir)
    img_idx = 0
    img_df = pd.DataFrame(columns = ('filename', 'img_index'), index = np.arange(0, numClasses*numImagesPerClass))
    
    for direc in train_dirs:
        img_dir_path = os.path.join('c:\home\classes\IE534_DL/hw5/src/data/tiny-imagenet-200/train', direc, 'images')
        images = fileList(img_dir_path)
        for img_fn in images:
            if img_idx % 1000 == 0:
                print('{} / {}'.format(img_idx, 100000))
            img_df.loc[img_idx] = [img_fn, img_idx]
            img_idx += 1

    return img_df

img_df = generateImgIdx()
img_df.to_csv('./hw5/src/img_df.csv')
