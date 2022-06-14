import time
from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import monai
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from resnet import generate_model

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.model = generate_model()
    
    def forward(self,x):

        x = self.model(x)

        return x


"""
mod = generate_model(model_depth=18,n_input_channels=1,n_classes=128) 

print(mod)
x = torch.rand(1,1,32,32,32)
x = mod(x)
print(x.shape)

"""