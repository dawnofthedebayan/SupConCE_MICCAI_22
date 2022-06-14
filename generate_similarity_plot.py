import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import torchmetrics
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn.functional as F

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import cv2 as cv
from utils import seed_everything,create_dir,str2bool
from torchsummary import summary

from dataloader import ParanasalContrastiveDataModule
from dataloader_compare import ParanasalComparisonDataModule
from dataloader_finetune import ParanasalFineTuneDataModule
from dataloader_test import ParanasalTestDataModule

from resnet import generate_model,ResNetDecoder_Inpainting
from model_pl import PL_MODEL_TEST
from sklearn.metrics import confusion_matrix

import pandas as pd 
import seaborn as sns

def get_similarity_score( features,neg_features):

    features = torch.tensor(features).cuda()
    neg_features = torch.tensor(neg_features).cuda()

    labels = torch.cat([torch.arange(features.shape[0]) for i in range(1)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    neg_mat_1 = neg_features

    positive_matrix   = torch.matmul(features, features.T)
    negative_matrix   = torch.matmul(features, neg_mat_1.T)
    
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    positive_matrix = positive_matrix[~mask].view(positive_matrix.shape[0], -1)
    negative_matrix = negative_matrix.view(negative_matrix.shape[0], -1)
        
            

    return positive_matrix,negative_matrix


def calculate_sim_score(folder):

    array_1  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_0std.npy")
    array_2  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_1std.npy")
    array_3  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_2std.npy")
    array_4  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_3std.npy")
    array_5  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_4std.npy")
    array_6  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_5std.npy")
    array_7  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_6std.npy")
    array_8  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_7std.npy")
    array_9  = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_8std.npy")
    array_10 = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_9std.npy")
    array_11 = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_10std.npy")
    array_12 = np.load(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{folder}/smax_11std.npy")

    positive_sim_score,negative_sim_score_1 = get_similarity_score(array_1,array_2)
    positive_sim_score,negative_sim_score_2 = get_similarity_score(array_1,array_3)
    positive_sim_score,negative_sim_score_3 = get_similarity_score(array_1,array_4)
    positive_sim_score,negative_sim_score_4 = get_similarity_score(array_1,array_5)
    positive_sim_score,negative_sim_score_5 = get_similarity_score(array_1,array_6)
    positive_sim_score,negative_sim_score_6 = get_similarity_score(array_1,array_7)
    positive_sim_score,negative_sim_score_7 = get_similarity_score(array_1,array_8)
    positive_sim_score,negative_sim_score_8 = get_similarity_score(array_1,array_9)
    positive_sim_score,negative_sim_score_9 = get_similarity_score(array_1,array_10)
    positive_sim_score,negative_sim_score_10 = get_similarity_score(array_1,array_11)
    positive_sim_score,negative_sim_score_11 = get_similarity_score(array_1,array_12)


    positive_sim_score = torch.flatten(positive_sim_score, start_dim=0, end_dim=- 1)
    negative_sim_score_1 = torch.flatten(negative_sim_score_1, start_dim=0, end_dim=- 1)
    negative_sim_score_2 = torch.flatten(negative_sim_score_2, start_dim=0, end_dim=- 1)
    negative_sim_score_3 = torch.flatten(negative_sim_score_3, start_dim=0, end_dim=- 1)
    negative_sim_score_4 = torch.flatten(negative_sim_score_4, start_dim=0, end_dim=- 1)
    negative_sim_score_5 = torch.flatten(negative_sim_score_5, start_dim=0, end_dim=- 1)
    negative_sim_score_6 = torch.flatten(negative_sim_score_6, start_dim=0, end_dim=- 1)
    negative_sim_score_7 = torch.flatten(negative_sim_score_7, start_dim=0, end_dim=- 1)
    negative_sim_score_8 = torch.flatten(negative_sim_score_8, start_dim=0, end_dim=- 1)
    negative_sim_score_9 = torch.flatten(negative_sim_score_9, start_dim=0, end_dim=- 1)
    negative_sim_score_10 = torch.flatten(negative_sim_score_10, start_dim=0, end_dim=- 1)
    negative_sim_score_11 = torch.flatten(negative_sim_score_11, start_dim=0, end_dim=- 1)

    return positive_sim_score,negative_sim_score_1,negative_sim_score_2,negative_sim_score_3,negative_sim_score_4,negative_sim_score_5,negative_sim_score_6,negative_sim_score_7,negative_sim_score_8,negative_sim_score_9,negative_sim_score_10,negative_sim_score_11



positive_sim_score,negative_sim_score_1,negative_sim_score_2,negative_sim_score_3,negative_sim_score_4,negative_sim_score_5,negative_sim_score_6,negative_sim_score_7,negative_sim_score_8,negative_sim_score_9,negative_sim_score_10,negative_sim_score_11 = calculate_sim_score("simclr")

total_samples = 1000
std_x_axis = ["0"] * total_samples 
std_x_axis = std_x_axis + ["1"] * total_samples
std_x_axis = std_x_axis + ["2"] * total_samples
std_x_axis = std_x_axis + ["3"] * total_samples
std_x_axis = std_x_axis + ["4"] * total_samples
std_x_axis = std_x_axis + ["5"] * total_samples
std_x_axis = std_x_axis + ["6"] * total_samples
std_x_axis = std_x_axis + ["7"] * total_samples
std_x_axis =  std_x_axis + ["8"] * total_samples
std_x_axis = std_x_axis + ["9"] * total_samples
std_x_axis = std_x_axis + ["10"] * total_samples
std_x_axis = std_x_axis + ["11"] * total_samples

method = ["simclr"] * total_samples * 12


sim_score = list(positive_sim_score.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_1.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_2.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_3.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_4.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_5.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_6.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_7.cpu().numpy())[:total_samples]
sim_score =  sim_score + list(negative_sim_score_8.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_9.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_10.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_11.cpu().numpy())[:total_samples]




positive_sim_score,negative_sim_score_1,negative_sim_score_2,negative_sim_score_3,negative_sim_score_4,negative_sim_score_5,negative_sim_score_6,negative_sim_score_7,negative_sim_score_8,negative_sim_score_9,negative_sim_score_10,negative_sim_score_11 = calculate_sim_score("ours")

total_samples = 1000
std_x_axis = std_x_axis + ["0"] * total_samples 
std_x_axis = std_x_axis + ["1"] * total_samples
std_x_axis = std_x_axis + ["2"] * total_samples
std_x_axis = std_x_axis + ["3"] * total_samples
std_x_axis = std_x_axis + ["4"] * total_samples
std_x_axis = std_x_axis + ["5"] * total_samples
std_x_axis = std_x_axis + ["6"] * total_samples
std_x_axis = std_x_axis + ["7"] * total_samples
std_x_axis =  std_x_axis + ["8"] * total_samples
std_x_axis = std_x_axis + ["9"] * total_samples
std_x_axis = std_x_axis + ["10"] * total_samples
std_x_axis = std_x_axis + ["11"] * total_samples

method = method + ["ours"] * total_samples * 12


sim_score = sim_score + list(positive_sim_score.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_1.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_2.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_3.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_4.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_5.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_6.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_7.cpu().numpy())[:total_samples]
sim_score =  sim_score + list(negative_sim_score_8.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_9.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_10.cpu().numpy())[:total_samples]
sim_score = sim_score + list(negative_sim_score_11.cpu().numpy())[:total_samples]


list_of_tuples = list(zip(std_x_axis, sim_score,method))
df = pd.DataFrame(list_of_tuples,
                  columns = ['Standard Deviation', 'Similarity Score','Method'])

sns_plot =  sns.lineplot(data=df, x="Standard Deviation", y="Similarity Score",hue="Method" )
plt.savefig('comparison.png')







