
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


class ParanasalDataset(Dataset):

    def __init__(self, dataset, dimension, transforms=None):
                
        self.dataset = dataset
        print("Processing {} datas".format(len(self.dataset)))
        self.ssl = ssl
        self.augmentation = augmentation  
        self.train = train
        self.idx = 0

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])

        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.dataset)


class ParanasalComparisonDataModule(pl.LightningDataModule):


    def __init__(self, root_dir, dimension, batch_size,ssl_type = "autoencode"):
        super().__init__()
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.dimension = dimension
        self.ssl_type = ssl_type
        #self.subjects = {"smax": [],"sphen":[],"sfront":[],"seth":[],"nose":[]}
        self.subjects = {"smax": []}
        self.test_subjects = None
        self.preprocess = None
        self.transform = None

        self.prepare_data()
       
    
    def get_max_shape(self, subjects):

        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)
    
    def __get__crop__(self,data,location,flip=False,std_factor= None):
    
        #Function to crop out sub volume
        cmin,cmax =  int(location["coronal"]["min"]["mean"]),int(location["coronal"]["max"]["mean"])
        smin,smax =  int(location["sagittal"]["min"]["mean"]),int(location["sagittal"]["max"]["mean"])
        amin,amax =  int(location["axial"]["min"]["mean"]),int(location["axial"]["max"]["mean"])

        if std_factor is not None: 

            cmin_std,cmax_std =  int(location["coronal"]["min"]["std"]),int(location["coronal"]["max"]["std"])
            smin_std,smax_std =  int(location["sagittal"]["min"]["std"]),int(location["sagittal"]["max"]["std"])
            amin_std,amax_std =  int(location["axial"]["min"]["std"]),int(location["axial"]["max"]["std"])

            c_width = cmax - cmin
            s_width = smax - smin
            a_width = amax - amin

            translation_direction_c = random.choice([-1,1])
            translation_direction_s = random.choice([-1,1])
            translation_direction_a = random.choice([-1,1])

            
            smin = int(smin + translation_direction_s*std_factor[0]*smin_std)
            smax = int(smin + s_width)

            cmin = int(cmin + translation_direction_c*std_factor[1]*cmin_std)
            cmax = int(cmin + c_width)

            amin = int(amin + translation_direction_a*std_factor[2]*amin_std)
            amax = int(amin + a_width)


        out = data[smin:smax,cmin:cmax,amin:amax]

        if flip:
            out =  np.array(np.flip(out, axis=0), dtype=np.float)

        y = 0
        if self.ssl_type =="rotate":
            
            out,y  =  rotate_batch_3d(out)

        elif self.ssl_type =="jigsaw":

            out = np.expand_dims(out,0)
            transform = tio.Resize((self.dimension, self.dimension,self.dimension))
            out = transform(out)
            out = np.squeeze(out)
            out,y =  jigsaw(
                                out,
                                4,
                                0,
                                is_training=False,
                                mode3d=True,
                            )

     
        out = np.expand_dims(out,axis=0)
        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(out)),label=y)
        return subject


    def prepare_data(self):
    
        normal_mri = [self.root_dir + "/normal/" + f for f in listdir(self.root_dir + "/normal") if isfile(join(self.root_dir + "/normal", f))]
        pathology_mri = [self.root_dir + "/pathology/" + f for f in listdir(self.root_dir + "/pathology") if isfile(join(self.root_dir + "/pathology", f))]
        unlabelled_mri = [self.root_dir + "/unlabelled/" + f for f in listdir(self.root_dir + "/unlabelled") if isfile(join(self.root_dir + "/unlabelled", f))]


        all_mri = np.array(normal_mri + pathology_mri + unlabelled_mri)
        #all_mri = all_mri[:10]
        np.random.shuffle(all_mri)
        print("Preparing Dataset...")

        for image_path in tqdm(all_mri,total= len(all_mri)):

            # 'image' and 'label' are arbitrary names for the images
            assert os.path.isfile(image_path)

            img = nibabel.load(image_path)
            #Crop Sinus Maxilliaris L
            smax_l = self.__get__crop__(img.get_data(),cfg["smax_l"])
            #Crop Sinus Maxilliaris R
            smax_r = self.__get__crop__(img.get_data(),cfg["smax_r"],flip=True)

            #Crop Sinus Maxilliaris L with pm 1 std
            smax_l_near = self.__get__crop__(img.get_data(),cfg["smax_l"],std_factor=(1,1,0.1))
            #Crop Sinus Maxilliaris L with pm 1 std
            smax_r_near = self.__get__crop__(img.get_data(),cfg["smax_r"],flip=True,std_factor=(1,1,0.1))


            #Crop Sinus Maxilliaris L with pm 1 std
            smax_l_near_2 = self.__get__crop__(img.get_data(),cfg["smax_l"],std_factor=(2,2,1))
            #Crop Sinus Maxilliaris L with pm 1 std
            smax_r_near_2 = self.__get__crop__(img.get_data(),cfg["smax_r"],flip=True,std_factor=(2,2,1))


            #Crop Sinus Maxilliaris L with pm 1 std
            smax_l_near_3 = self.__get__crop__(img.get_data(),cfg["smax_l"],std_factor=(3,3,1))
            #Crop Sinus Maxilliaris L with pm 1 std
            smax_r_near_3 = self.__get__crop__(img.get_data(),cfg["smax_r"],flip=True,std_factor=(3,3,1))

            

            self.subjects["smax"].append(smax_l)
            self.subjects["smax"].append(smax_r)
            self.subjects["smax"].append(smax_l_near)
            self.subjects["smax"].append(smax_r_near)
            self.subjects["smax"].append(smax_l_near_2)
            self.subjects["smax"].append(smax_r_near_2)
            self.subjects["smax"].append(smax_l_near_3)
            self.subjects["smax"].append(smax_r_near_3)

            #data = np.squeeze(smax_l)
            #new_image = nibabel.Nifti1Image(data.astype(np.float), affine=np.eye(4))
            #nibabel.save(new_image,f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/my_code/dummy/jigsaw.nii.gz")

           
    
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
        self.smax = tio.SubjectsDataset(self.subjects["smax"], transform=self.transform)


        
    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        
        loader = {

            "smax" :DataLoader(self.smax, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
            
        }

        return loader


if __name__ == '__main__':

    p = ParanasalContrastiveDataModule("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset",32, 8, 0.8)

    p.prepare_data()