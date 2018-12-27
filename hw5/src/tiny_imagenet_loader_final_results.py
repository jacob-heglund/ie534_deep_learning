from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from random import randint
from PIL import Image

class tinyImageNetDataset(Dataset):
    def __init__(self, src_dir, data_dir, annotations_path, train = True, final_results = False, transform = None):
        self.transform = transform
        self.train = train
        self.numClasses = 200
        self.numImagesPerClass = 500
        
        self.src_dir = src_dir
        self.data_dir = data_dir
        # this reads from a tab-separated txt file
        self.val_img_df = pd.read_csv(annotations_path, delimiter = '\t')
        self.val_img_df.columns = ["filename", "class", "pt1", "pt2", "pt3", "pt4"]

    def __len__(self):
        if self.train:
            return 100000
        else:
            return 10000

    def __getitem__(self, idx):
        # return an image and its label from the validation set
        # for idx, get the class label
        fn = self.val_img_df['filename'][idx]
        label = str(self.val_img_df['class'][idx])
        
        # get the image filepath and return PIL image
        val_img_path = os.path.join(self.src_dir, self.data_dir, 'images', fn)

        val_img = Image.open(val_img_path).convert('RGB')
        if self.transform is not None:
            val_img = self.transform(val_img)
        
        return val_img, label


    def fileList(self, rootDir):
        '''generate a list of all filenames within a directory'''
        fileList = []
        for files in next(os.walk(rootDir)):
            if files != []:
                fileList.append(files)
        return fileList[1]



        