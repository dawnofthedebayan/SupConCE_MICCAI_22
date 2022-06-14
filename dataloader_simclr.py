
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

cfg = {
        "smax_l": {"coronal":  { "min": {"mean":151,"std":1.899835519},
                                "max": {"mean":198.5,"std":1.414213562}
                                },

                 "sagittal":  { "min": {"mean":39.5,"std":1.322875656},
                                 "max": {"mean": 75.75,"std":1.785357107}
                                },

                 "axial":      { "min": {"mean":68.875,"std":1.964529206},
                                 "max": {"mean": 113.5,"std":1.802775638}
                                }
                 },

        "smax_r": {"coronal":  { "min": {"mean":151,"std":2.175861898},
                                "max": {"mean":198.375,"std":1.316956719}
                                },

                 "sagittal":  { "min": {"mean":95.25,"std":1.71391365},
                                 "max": {"mean": 128.875,"std":2.315032397}
                                },

                 "axial":      { "min": {"mean":66.375,"std":6.479535091},
                                 "max": {"mean": 111.5,"std":7.465145348}
                                }
                 },

        "sphen": {"coronal":  { "min": {"mean":123.75,"std":7.066647013},
                                "max": {"mean":158.375,"std":4.370036867}
                                },

                 "sagittal":  { "min": {"mean":63.625,"std":3.533323506},
                                 "max": {"mean": 103.875,"std":4.0754601}
                                },

                 "axial":      { "min": {"mean":99.625,"std":2.446298224},
                                 "max": {"mean": 127.625,"std":2.287875652}
                                }
                 },

        "sfront": {"coronal":  { "min": {"mean":185,"std":2.618614683},
                                "max": {"mean":208.2857143,"std":1.829464068}
                                },

                 "sagittal":  { "min": {"mean":54.14285714,"std":8.773801447},
                                 "max": {"mean": 109.4285714,"std":10.18201696}
                                },

                 "axial":      { "min": {"mean":126,"std":4.035556255},
                                 "max": {"mean": 156.8571429,"std":6.685347975}
                                }
                 },


        "seth": {"coronal":  { "min": {"mean":152.5714286,"std":2.258769757},
                                "max": {"mean":197.7142857,"std":4.025429372}
                                },

                 "sagittal":  { "min": {"mean":71.57142857,"std":9.897433186},
                                 "max": {"mean":101.8571429,"std":1.456862718}
                                },

                 "axial":      { "min": {"mean":104.5714286,"std":1.916629695},
                                 "max": {"mean": 129.8571429,"std":3.090472522}
                                }
                 },


        "nose": {"coronal":  { "min": {"mean":147.3333333,"std":4.229525847},
                                "max": {"mean":201.6666667,"std":2.924988129}
                                },

                 "sagittal":  { "min": {"mean":68.5,"std":1.802775638},
                                 "max": {"mean":99.33333333,"std":1.885618083}
                                },

                 "axial":      { "min": {"mean":73.16666667,"std":3.89087251},
                                 "max": {"mean": 123.8333333,"std":2.477678125}
                                }
                 },
        
      }




class ParanasalSimCLRDataModule(pl.LightningDataModule):


    def __init__(self, root_dir,dimension, batch_size,trainset,valset,testset,augmentations=None):
        super().__init__()
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.dimension = dimension
        self.subjects = {"train_cls_1": [],"train_cls_2": [],"val": [],"test":[]}
        self.preprocess = None
        self.augmentations = augmentations
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
            if ds_type == "train": 
                if label == 0:
                    
                    
                    self.subjects[ds_type + "_cls_1"].append(volume)

                else:
                    
                    self.subjects[ds_type + "_cls_2"].append(volume)

            else:

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
    
    

    def setup(self, stage=None):

        
        self.preprocess = self.get_preprocessing_transform()
        self.transform = tio.Compose([self.preprocess])

        #self.val   = tio.SubjectsDataset(self.subjects["val"],   transform=self.transform)
        self.val   = tio.SubjectsDataset(self.subjects["train_cls_1"] + self.subjects["train_cls_2"],   transform=self.transform)
        self.test  = tio.SubjectsDataset(self.subjects["test"],  transform=self.transform)

        self.train_aug = None

        if len(self.augmentations) > 0 :

            print("AUGMENTATING TRAINING SET")
            self.transform = tio.Compose([self.preprocess,tio.Compose(self.augmentations)])

        else:
            print("NO AUGMENTATION")
        
        self.train_cls_1_v1 = tio.SubjectsDataset(self.subjects["train_cls_1"], transform=self.transform)
        self.train_cls_1_v2 = tio.SubjectsDataset(self.subjects["train_cls_1"], transform=self.transform)
        self.train_cls_2_v1 = tio.SubjectsDataset(self.subjects["train_cls_2"], transform=self.transform)
        self.train_cls_2_v2 = tio.SubjectsDataset(self.subjects["train_cls_2"], transform=self.transform)


    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        loader = {

            "cls_1_v1":DataLoader(self.train_cls_1_v1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False),
            "cls_2_v1":DataLoader(self.train_cls_2_v1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False),
            "cls_1_v2":DataLoader(self.train_cls_1_v2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False),
            "cls_2_v2":DataLoader(self.train_cls_2_v2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)
        
        }
        return loader
        
    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)


if __name__ == '__main__':

    p = ParanasalContrastiveDataModule("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset",32, 8, 0.8)

    p.prepare_data()