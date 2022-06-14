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
from dataloader_simclr import ParanasalSimCLRDataModule
from dataloader_compare import ParanasalComparisonDataModule
from dataloader_finetune import ParanasalFineTuneDataModule
from dataloader_test import ParanasalTestDataModule

from resnet import generate_model,ResNetDecoder_Inpainting
from model_pl import PL_MODEL,PL_MODEL_SIMCLR,PL_MODEL_COMPARE,PL_MODEL_FT,PL_MODEL_TEST
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=50, help='epoch number')

    parser.add_argument('--batch_size', type=int,
                        default=256, help='epoch number')
    
    parser.add_argument('--root_dir', type=str,
                        default="/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/", help='Source Folder')

    parser.add_argument('--save_root_folder', type=str,
                        default="/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/", help='Checkpoints')

    parser.add_argument('--seed', type=int,
                        default=58, help='Checkpoints')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='Checkpoints')

    parser.add_argument('--n_classes', type=int,
                        default=10, help='Checkpoints')

    parser.add_argument('--model', type=str,
                        default='unet', help='Checkpoints')

    parser.add_argument('--expt_name', type=str,
                        default='ours_cv_1', help='Checkpoints')

    parser.add_argument('--model_depth', type=int,
                        default=18, help='Checkpoints')

    parser.add_argument('--latent_dim', type=int,
                        default=128, help='Checkpoints')
    
    parser.add_argument('--dimension', type=int,
                        default=32, help='Checkpoints')

    parser.add_argument('--finetune', type=str2bool,
                        default="False", help='Checkpoints')

    parser.add_argument('--ckpt_path', type=str,
                        default="/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/checkpoint/SSL_pretrained_weights/rotate/model.ckpt", help='Checkpoints')

    parser.add_argument('--ssl_type', type=str,
                        default="ours", help='Checkpoints')

    parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

    parser.add_argument('--train_percentage', default=0.9, type=float,
                    help='softmax temperature (default: 0.07)')

    opt = parser.parse_args()

    seed_everything(opt.seed)

    encoder = generate_model(model_depth=opt.model_depth,n_input_channels=1,n_classes=opt.latent_dim)

    if opt.finetune: 

        dm = ParanasalFineTuneDataModule(root_dir=opt.root_dir,dimension=opt.dimension,batch_size=opt.batch_size,train_val_split=opt.train_percentage)
        pl_model = PL_MODEL_FT(model=encoder,opt=opt,ckpt_path=opt.ckpt_path)

    
    else: 
        if opt.ssl_type == "simclr":
            dm = ParanasalSimCLRDataModule(root_dir=opt.root_dir,dimension=opt.dimension,batch_size=opt.batch_size)
            pl_model = PL_MODEL_SIMCLR(model=encoder,opt=opt)

        elif opt.ssl_type == "rotate" or opt.ssl_type == "jigsaw":

            print("SSL PRETRAIN:",opt.ssl_type)
            dm = ParanasalComparisonDataModule(root_dir=opt.root_dir,dimension=opt.dimension,batch_size=opt.batch_size,ssl_type=opt.ssl_type)
            pl_model = PL_MODEL_COMPARE(model=encoder,opt=opt)

        elif opt.ssl_type == "autoencode":
            
            print("SSL PRETRAIN:",opt.ssl_type)
            decoder = ResNetDecoder_Inpainting(autoencoding=True)
            dm = ParanasalComparisonDataModule(root_dir=opt.root_dir,dimension=opt.dimension,batch_size=opt.batch_size,ssl_type=opt.ssl_type)
            pl_model = PL_MODEL_COMPARE(model=encoder,opt=opt,decoder=decoder)
        else:
            dm = ParanasalTestDataModule(root_dir=opt.root_dir,dimension=opt.dimension,batch_size=opt.batch_size)
            pl_model = PL_MODEL_TEST(model=encoder,opt=opt).load_from_checkpoint(model=encoder,opt=opt,
                                    checkpoint_path= opt.ckpt_path)


    trainer = Trainer(
            max_epochs =opt.epoch,
            gpus=1,
            logger=None,
            limit_train_batches=1.0,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            checkpoint_callback=False,
            reload_dataloaders_every_epoch=True,
            num_sanity_val_steps=1,  # Skip Sanity Check
        )
        
    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_0std"
    dm.ds_to_return = "smax_0std"
    trainer.test(pl_model, datamodule=dm)

    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_1std"
    dm.ds_to_return = "smax_1std"
    trainer.test(pl_model, datamodule=dm)

    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_2std"
    dm.ds_to_return = "smax_2std"
    trainer.test(pl_model, datamodule=dm)

    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_3std"
    dm.ds_to_return = "smax_3std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_4std"
    dm.ds_to_return = "smax_4std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_5std"
    dm.ds_to_return = "smax_5std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_6std"
    dm.ds_to_return = "smax_6std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_7std"
    dm.ds_to_return = "smax_7std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_8std"
    dm.ds_to_return = "smax_8std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_9std"
    dm.ds_to_return = "smax_9std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_10std"
    dm.ds_to_return = "smax_10std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_11std"
    dm.ds_to_return = "smax_11std"
    trainer.test(pl_model, datamodule=dm)


    pl_model.feature_array = None
    pl_model.ds_to_save = "smax_12std"
    dm.ds_to_return = "smax_12std"
    trainer.test(pl_model, datamodule=dm)



    #if opt.finetune: trainer.test(ckpt_path="best")
  