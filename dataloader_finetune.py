
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




class ParanasalFineTuneDataModule(pl.LightningDataModule):


    def __init__(self, root_dir, dimension, batch_size,ssl_type = "autoencode",train_val_split=0.9):
        super().__init__()
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.dimension = dimension
        self.ssl_type = ssl_type
        #self.subjects = {"smax": [],"sphen":[],"sfront":[],"seth":[],"nose":[]}
        self.subjects = {"train": [],"val": [],"test": []}
        self.preprocess = None
        self.transform = None

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

    def __prepare_dataset__(self,data_list,label,ds_type):

        for image_path in tqdm(data_list,total= len(data_list)):
            # 'image' and 'label' are arbitrary names for the images
            assert os.path.isfile(image_path)
            img = nibabel.load(image_path)
            #Crop Sinus Maxilliaris L
            volume = self.__get__data__(img.get_data(),label)
            self.subjects[ds_type].append(volume)


    def prepare_data(self):
        
        class_1 = [self.root_dir + "/roi_dataset/1/smax/" + f for f in listdir(self.root_dir + "/roi_dataset/1/smax") if isfile(join(self.root_dir + "/roi_dataset/1/smax", f))]
        class_2 = [self.root_dir + "/roi_dataset/2/smax/" + f for f in listdir(self.root_dir + "/roi_dataset/2/smax") if isfile(join(self.root_dir + "/roi_dataset/2/smax", f))]
        class_3 = [self.root_dir + "/roi_dataset/3/smax/" + f for f in listdir(self.root_dir + "/roi_dataset/3/smax") if isfile(join(self.root_dir + "/roi_dataset/3/smax", f))]


        #np.random.shuffle(class_1)
        #np.random.shuffle(class_2)
        #np.random.shuffle(class_3)


        class_1_test = class_1[int(len(class_1)*0.5):]
        train_val_class_1 = class_1[:int(len(class_1)*self.train_val_split)]

        class_2_test = class_2[int(len(class_2)*0.5):]
        train_val_class_2 = class_2[:int(len(class_2)*self.train_val_split)]

        class_3_test = class_3[int(len(class_3)*0.5):]
        train_val_class_3 = class_3[:int(len(class_3)*self.train_val_split)]



        


        #class_1_train = train_val_class_1[:int(len(train_val_class_1)*0.9)]
        #class_2_train = train_val_class_2[:int(len(train_val_class_2)*0.9)]
        #class_3_train = train_val_class_3[:int(len(train_val_class_3)*0.9)]
        class_1_train = train_val_class_1
        class_2_train = train_val_class_2
        class_3_train = train_val_class_3

        #class_1_val = train_val_class_1[int(len(train_val_class_1)*0.9):int(len(train_val_class_1))]
        #class_2_val = train_val_class_2[int(len(train_val_class_2)*0.9):int(len(train_val_class_2))]
        #class_3_val = train_val_class_3[int(len(train_val_class_3)*0.9):int(len(train_val_class_3))]

        class_1_val = class_1_test
        class_2_val = class_2_test
        class_3_val = class_3_test

        print(len(class_1_train),len(class_1_val),len(class_1_test))
        print(len(class_2_train),len(class_2_val),len(class_2_test))
        print(len(class_3_train),len(class_3_val),len(class_3_test))

        print("Preparing Dataset...")
        self.__prepare_dataset__(class_1_train,0,"train")
        self.__prepare_dataset__(class_2_train,1,"train")
        self.__prepare_dataset__(class_3_train,2,"train")

        self.__prepare_dataset__(class_1_val,0,"val")
        self.__prepare_dataset__(class_2_val,1,"val")
        self.__prepare_dataset__(class_3_val,2,"val")

        self.__prepare_dataset__(class_1_test,0,"test")
        self.__prepare_dataset__(class_2_test,1,"test")
        self.__prepare_dataset__(class_3_test,2,"test")
        

           
    
    def get_preprocessing_transform(self):

        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.Resize((self.dimension, self.dimension,self.dimension)),
            tio.EnsureShapeMultiple(8),
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):

        
        self.preprocess = self.get_preprocessing_transform()
        self.transform = tio.Compose([self.preprocess])
        
        self.train = tio.SubjectsDataset(self.subjects["train"], transform=self.transform)
        self.val = tio.SubjectsDataset(self.subjects["val"], transform=self.transform)
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