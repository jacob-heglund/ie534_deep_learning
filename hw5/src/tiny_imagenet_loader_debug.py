from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from random import randint
from PIL import Image

class tinyImageNetDatasetDebug(Dataset):
    def __init__(self, src_dir, data_dir, annotations_path, transform):
        self.src_dir = src_dir
        self.transform = transform

        self.data_dir = data_dir
        # this reads from a csv file
        self.train_img_df = pd.read_csv(annotations_path)
        self.numClasses = 200
        self.numImagesPerClass = 500

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        # idx is a unique identifying number for each image
        # return a triplet (query, positive, negative) from training set
        # get query image
        query_img_fn = self.train_img_df['filename'][idx]
        pos_dir = query_img_fn[0:9]
        query_img_path = os.path.join(self.src_dir, self.data_dir, pos_dir, 'images', query_img_fn)
        pos_class_path = os.path.join(self.src_dir, self.data_dir, pos_dir, 'images')
        images = self.fileList(pos_class_path)

        # get positive image
        sample_pos_img = True
        while sample_pos_img:
                pos_rand_img_fn = images[randint(0, self.numImagesPerClass-1)]
                if pos_rand_img_fn != query_img_fn:
                    pos_img_fn = pos_rand_img_fn
                    pos_img_path = os.path.join(pos_class_path, pos_img_fn)
                    sample_pos_img = False
        
        # get negative image
        sample_neg_dir = True
        while sample_neg_dir:
            neg_dir_idx = randint(0, (self.numClasses * self.numImagesPerClass) -1)
            neg_dir = self.train_img_df['filename'][neg_dir_idx][0:9]
            if neg_dir != pos_dir:
                sample_neg_dir = False
                neg_class_path = os.path.join(self.src_dir, self.data_dir, neg_dir, 'images')
                neg_images = self.fileList(neg_class_path)
                neg_img_fn = neg_images[randint(0, self.numImagesPerClass-1)] 
                neg_img_path = os.path.join(neg_class_path, neg_img_fn)

        # return PIL images (some are greyscale, so convert to RGB)
        query_img = Image.open(query_img_path).convert('RGB')
        pos_img = Image.open(pos_img_path).convert('RGB')
        neg_img = Image.open(neg_img_path).convert('RGB')
        #triplet_path = [query_img_path, pos_img_path, neg_img_path]


        # apply transforms
        if self.transform is not None:
            query_img, pos_img, neg_img = self.transform(query_img), self.transform(pos_img), self.transform(neg_img)

        triplet = [query_img, pos_img, neg_img]
        query_class = pos_dir
        return triplet, query_class


    def fileList(self, rootDir):
        '''generate a list of all filenames within a directory'''
        fileList = []
        for files in next(os.walk(rootDir)):
            if files != []:
                fileList.append(files)
        return fileList[1]



        