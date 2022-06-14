import torch
import numpy as np
from thop import profile
from thop import clever_format
import os 
import random 
import torch.nn.functional as F
import torch.nn as nn

from typing import Tuple
import torchio as tio
import random

class RandomCrop:
    """Random cropping on subject."""

    def __init__(self, roi_size: Tuple,p=0.5):
        """Init.

        Args:
            roi_size: cropping size.
        """
        self.sampler = tio.data.LabelSampler(patch_size=roi_size, label_name="label")
        self.p = p

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Use patch sampler to crop.

        Args:
            subject: subject having image and label.

        Returns:
            cropped subject
        """
        if random.random() < self.p: 
            for patch in self.sampler(subject=subject, num_patches=1):
                return patch
        else:
            return subject

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_augmentations(opt): 

    augment_types = opt.augment.split(",")

    augment_fns = []

    for aug in augment_types:
        if aug == "randcrop":
            print("RANDOM CROP AUGMENTATION")
            patch_size = int(0.8*opt.dimension) 
            augment_fns.append(RandomCrop((patch_size,patch_size,patch_size),p=0.5))
            augment_fns.append(tio.Resize((opt.dimension, opt.dimension,opt.dimension)))

        elif aug == "noise":
            print("NOISE AUGMENTATION")
            augment_fns.append(tio.RandomNoise(p=0.5))

        elif aug == "affine":
            print("AFFINE AUGMENTATION")
            augment_fns.append(tio.RandomAffine(
                                scales=(0.8, 1.3),
                                degrees=15, p=0.5
                            ))

        elif aug == "elastic":
            print("ELASTIC AUGMENTATION")
            augment_fns.append(tio.RandomElasticDeformation(
                                num_control_points=5, 
                                max_displacement=5,
                                locked_borders=2,p=0.5))

        elif aug == "flip":
            print("FLIP AUGMENTATION")
            augment_fns.append(tio.RandomFlip(axes=0, p=0.5))

    return augment_fns

    

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def rotate_batch_3d(x, y=None):
        
        
    
    #x = np.expand_dims(x,axis =0)
    volume  = np.squeeze(x)
    #print(x.shape)
    y = np.zeros((10))
    
    rotated_batch = []
    #np.random.seed(np.random.randint(99999))
    rot = np.random.randint(10) #- 1
    
    if rot == 0:
        volume = volume
    elif rot == 1:
        #volume = np.transpose(np.flip(volume, 1), (1, 0, 2, 3))  # 90 deg Z
        volume = np.rot90(volume,k=1,axes=(1,2))  # 90 deg Z
    elif rot == 2:
        #volume = np.flip(volume, (0, 1))  # 180 degrees on z axis
        volume = np.rot90(volume,k=2,axes=(1,2))
    elif rot == 3:
        #volume = np.flip(np.transpose(volume, (1, 0, 2, 3)), 1)  # 90 deg Z
        volume = np.rot90(volume,k=3,axes=(1,2))
    elif rot == 4:
        #volume = np.transpose(np.flip(volume, 1), (0, 2, 1, 3))  # 90 deg X
        volume = np.rot90(volume,k=1,axes=(0,2))
    elif rot == 5:
        #volume = np.flip(volume, (1, 2))  # 180 degrees on x axis
        volume = np.rot90(volume,k=2,axes=(0,2))
    elif rot == 6:
        #volume = np.flip(np.transpose(volume, (0, 2, 1, 3)), 1)  # 90 deg X
        volume = np.rot90(volume,k=3,axes=(0,2))
    elif rot == 7:
        #volume = np.transpose(np.flip(volume, 0), (2, 1, 0, 3))  # 90 deg Y
        volume = np.rot90(volume,k=1,axes=(0,1))
    elif rot == 8:
        #volume = np.flip(volume, (0, 2))  # 180 degrees on y axis
        volume = np.rot90(volume,k=2,axes=(0,1))
    elif rot == 9:
        #volume = np.flip(np.transpose(volume, (2, 1, 0, 3)), 0)  # 90 deg Y
        volume = np.rot90(volume,k=3,axes=(0,1))

    #rotated_batch.append(volume)
    y[rot] = 1

    return volume.copy(), rot



def load_permutations_3d(
        permutation_path="/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/my_code/permutations3d_100_64.npy"):
    with open(permutation_path, "rb") as f:
        perms = np.load(f)

    return perms, len(perms)




def crop_patches_3d(image, is_training, patches_per_side, patch_jitter=0):

    h, w, d = image.shape

    patch_overlap = -patch_jitter if patch_jitter < 0 else 0

    h_grid = (h - patch_overlap) // patches_per_side
    w_grid = (w - patch_overlap) // patches_per_side
    d_grid = (d - patch_overlap) // patches_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter
    d_patch = d_grid - patch_jitter

    patches = []
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            for k in range(patches_per_side):

                p = do_crop_3d(image,
                            i * h_grid,
                            j * w_grid,
                            k * d_grid,
                            h_grid + patch_overlap,
                            w_grid + patch_overlap,
                            d_grid + patch_overlap)

                if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid:
                    
                    p = crop_3d(p, is_training, [h_patch, w_patch, d_patch])
                patches.append(p)
    return patches


def crop_3d(image, is_training, crop_size):
    h, w, d = crop_size[0], crop_size[1], crop_size[2]
    h_old, w_old, d_old = image.shape[0], image.shape[1], image.shape[2]

    if is_training:
        # crop random
        x = np.random.randint(0, 1+h_old-h)
        y = np.random.randint(0, 1+w_old-w)
        z = np.random.randint(0, 1+d_old-d)
    else:
        # crop center
        x = int((h_old - h) / 2)
        y = int((w_old - w) / 2)
        z = int((d_old - d) / 2)

    return do_crop_3d(image, x, y, z, h, w, d)


def do_crop_3d(image, x, y, z, h, w, d):

    assert type(x) == int, x
    assert type(y) == int, y
    assert type(z) == int, z
    assert type(h) == int, h
    assert type(w) == int, w
    assert type(d) == int, d

    return image[x:x + h, y:y + w, z:z + d]

def preprocess_image(image, is_training, patches_per_side, patch_jitter, permutations,mode3d):

    label = random.randint(0, len(permutations) - 1)

    
    patches = crop_patches_3d(image, is_training, patches_per_side, patch_jitter)
 
    b = np.zeros((len(permutations),))
    b[label] = 1
        
    return np.array(patches)[np.array(permutations[label])], np.array(b)

def unpatchify(patches,org_vol):

    data = np.zeros(org_vol.shape)
    counter = 0
    for k in range(0,data.shape[0],patches.shape[1]): 
        for j in range(0,data.shape[1],patches.shape[2]): 
            for i in range(0,data.shape[2],patches.shape[3]): 
                data[i:i+patches.shape[1],j:j+patches.shape[2],k:k+patches.shape[3]] = patches[counter]
                counter = counter + 1
             
    return data          


def jigsaw(volume, patches_per_side, patch_jitter, is_training=True, mode3d=True):
    xs = []
    ys = []

    permutations,_ = load_permutations_3d()

    
    
    x, y = preprocess_image(volume, is_training, patches_per_side, patch_jitter, permutations, mode3d)
    xs.append(x)
    ys.append(y)

    
    xs = np.squeeze(xs)
    data = unpatchify(xs,volume)

    ys = np.squeeze(np.stack(ys))
    ys = np.argmax(ys)

    return data, ys
