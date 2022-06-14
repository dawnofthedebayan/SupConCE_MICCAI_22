
import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split, DataLoader
import monai
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel 
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import random
from tqdm import tqdm
from utils import rotate_batch_3d,jigsaw




class ParanasalFineTuneDataModuleCV(pl.LightningDataModule):


    def __init__(self, root_dir, dimension, batch_size,trainset,valset,testset,ssl_type = "autoencode",augmentations=None ):

        super().__init__()
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.dimension = dimension
        self.ssl_type = ssl_type
        self.augmentations = augmentations
        self.subjects = {"train": [],"val": [],"test": []}
        self.preprocess = None
        self.transform = None
        self.trainset = trainset 
        self.valset = valset 
        self.testset = testset 

        self.prepare_data()
       
    
    def get_max_shape(self, subjects):

        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)
    
    def __get__data__(self,data,y):
    
        out = data 
        out = np.expand_dims(out,axis=0)
        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(out)),label=y)
        return subject

    def __prepare_dataset__(self,data_list,ds_type):
        
        for image_path,label in tqdm(data_list,total= len(data_list)):
            # 'image' and 'label' are arbitrary names for the images
            assert os.path.isfile(image_path)
            img = nibabel.load(image_path)
            #Crop Sinus Maxilliaris L
            volume = self.__get__data__(img.get_data(),label)
            self.subjects[ds_type].append(volume)


    def prepare_data(self):
            
        print("Preparing Dataset...")

        self.__prepare_dataset__(self.trainset,"train")
        self.__prepare_dataset__(self.valset,"val")
        self.__prepare_dataset__(self.testset,"test")
                   
    
    def get_preprocessing_transform(self):

        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.Resize((self.dimension, self.dimension,self.dimension)),
            tio.EnsureShapeMultiple(8),
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        return self.augmentations
       

    def setup(self, stage=None):

        
        self.preprocess = self.get_preprocessing_transform()
        self.transform = tio.Compose([self.preprocess])

        if len(self.augmentations) > 0:
            self.transform = tio.Compose([self.preprocess])
        else:
            print("APPLIED AUGMENTATIONS")
            self.transform = tio.Compose([self.preprocess,tio.Compose(self.augmentations)])


        
        self.train = tio.SubjectsDataset(self.subjects["train"], transform=self.transform)
        #self.val = tio.SubjectsDataset(self.subjects["val"], transform=self.transform)
        self.val = tio.SubjectsDataset(self.subjects["train"], transform=self.transform)
        self.test = tio.SubjectsDataset(self.subjects["test"], transform=self.transform)


        
    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        
        loader = {

            "train" :DataLoader(self.train, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
            
        }

        return loader


    def val_dataloader(self):
    

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)

    
    def test_dataloader(self):
    
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)


if __name__ == '__main__':

    p = ParanasalContrastiveDataModule("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset",32, 8, 0.8)

    p.prepare_data()