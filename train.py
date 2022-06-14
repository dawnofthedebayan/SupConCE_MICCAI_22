

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
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import cv2 as cv
from utils import seed_everything,create_dir,str2bool,RandomCrop,get_augmentations
from torchsummary import summary
from sklearn.utils import class_weight
from dataloader import ParanasalContrastiveDataModule
from dataloader_simclr import ParanasalSimCLRDataModule
from dataloader_compare import ParanasalComparisonDataModule
from dataloader_finetune import ParanasalFineTuneDataModule
from dataloader_finetune_cvexpt import ParanasalFineTuneDataModuleCV
import torchio as tio


from resnet import generate_model,ResNetDecoder_Inpainting
from model_pl import PL_MODEL,PL_MODEL_SIMCLR,PL_MODEL_COMPARE,PL_MODEL_FT,PL_MODEL_2,PL_MODEL_FT_2


from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=160, help='epoch number')

    parser.add_argument('--batch_size', type=int,
                        default=64, help='epoch number')
    
    parser.add_argument('--root_dir', type=str,
                        default="/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/", help='Source Folder')

    parser.add_argument('--save_root_folder', type=str,
                        default="/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/", help='Checkpoints')

    parser.add_argument('--seed', type=int,
                        default=58, help='Checkpoints')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='Checkpoints')

    parser.add_argument('--n_classes', type=int,
                        default=4, help='Checkpoints')

    parser.add_argument('--model', type=str,
                        default='unet', help='Checkpoints')

    parser.add_argument('--expt_name', type=str,
                        default='ours', help='Checkpoints')

    parser.add_argument('--model_depth', type=int,
                        default=18, help='Checkpoints')

    parser.add_argument('--latent_dim', type=int,
                        default=128, help='Checkpoints')
    
    parser.add_argument('--dimension', type=int,
                        default=32, help='Checkpoints')

    parser.add_argument('--finetune', type=str2bool,
                        default="True", help='Checkpoints')

    parser.add_argument('--ckpt_path', type=str,
                        default="/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/checkpoint/SSL_pretrained_weights/pretrain_ours_3/model_enc.ckpt", help='Checkpoints')

    parser.add_argument('--ssl_type', type=str,
                        default="autoencode", help='Checkpoints')

    parser.add_argument('--temperature', default=0.1, type=float,
                    help='softmax temperature (default: 0.07)')

    parser.add_argument('--train_percentage', default=1.0, type=float,
                    help='softmax temperature (default: 0.07)')

    parser.add_argument('--NUM_TRIALS', default=1, type=int,
                    help='softmax temperature (default: 0.07)')


    parser.add_argument('--std', default=1, type=int,
                    help='softmax temperature (default: 0.07)')

    parser.add_argument('--losses', default=1, type=int,
                    help='softmax temperature (default: 0.07)')

    parser.add_argument('--augment', default="", type=str,
                    help='softmax temperature (default: 0.07)')

    parser.add_argument('--dataset', default="roi_dataset_2", type=str,
                    help='softmax temperature (default: 0.07)')

    

    opt = parser.parse_args()

    seed_everything(opt.seed)

   
    #encoder = generate_model(model_depth=opt.model_depth,n_input_channels=1,n_classes=opt.latent_dim)

    

    class_good = [opt.root_dir + f"/{opt.dataset}/good/" + f for f in listdir(opt.root_dir + f"/{opt.dataset}/good/") if isfile(join(opt.root_dir + f"/{opt.dataset}/good/", f))]
    class_bad = [opt.root_dir + f"/{opt.dataset}/bad/" + f for f in listdir(opt.root_dir + f"/{opt.dataset}/bad/") if isfile(join(opt.root_dir + f"/{opt.dataset}/bad/", f))]

    image_paths = np.array(class_good + class_bad)
    labels = np.array([0]*len(class_good) + [1]*len(class_bad))
    
    for i in range(opt.NUM_TRIALS):

        outer_fold = 0 
        inner_fold = 0 
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.seed)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.seed)

        for train_index, test_index in outer_cv.split(image_paths, labels):
            outer_fold += 1
                           
            X_train_val, X_test = image_paths[train_index], image_paths[test_index]
            y_train_val, y_test = labels[train_index], labels[test_index]
            print(len(X_test)) 
            for train_index, val_index in inner_cv.split(X_train_val, y_train_val):
                inner_fold += 1
                
                X_train, X_val = X_train_val[train_index], X_train_val[val_index]
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]

                print(len(X_val),len(X_train)) 
                class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(y_train),
                                                y_train)
                

                print(f"\n\nEXPERIMENT OUTER FOLD {outer_fold} INNER FOLD {inner_fold}\n\n")
                
                trainset = [(x,y) for x,y in zip(X_train,y_train)]
                valset   = [(x,y) for x,y in zip(X_val,y_val)]
                testset   = [(x,y) for x,y in zip(X_test,y_test)]

                if opt.finetune: 

                    augmentations = get_augmentations(opt)
                    encoder = generate_model(model_depth=opt.model_depth,n_input_channels=1,n_classes=opt.latent_dim,include_fc=False)
                    dm = ParanasalFineTuneDataModuleCV(root_dir=opt.root_dir,dimension=opt.dimension,batch_size=opt.batch_size,trainset=trainset,valset=valset,testset=testset,augmentations=augmentations)
                    pl_model = PL_MODEL_FT_2(model=encoder,opt=opt,ckpt_path=opt.ckpt_path,cls_wt=torch.tensor(class_weights),expt_name=f"{opt.expt_name}_{opt.seed}_{inner_fold}",result_txt_name=f"{opt.expt_name}")

                    checkpoint_callback = ModelCheckpoint(
                        monitor="val/loss_epoch",
                        dirpath=f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/checkpoint/cv_expts/{opt.expt_name}/",
                        filename=f"model_{opt.seed}_{inner_fold}",
                        mode="min",
                        verbose=True,
                    )

                    early_stop_callback = EarlyStopping(monitor="val/loss_epoch", min_delta=0.00, patience=10, verbose=True, mode="min")
                    wandb_logger = WandbLogger(name=f"{opt.expt_name}_{opt.seed}_{inner_fold}",project="MRI-JournalOfRhinology")
                    trainer = Trainer(
                        max_epochs =opt.epoch,
                        gpus=1,
                        logger=None,
                        limit_train_batches=opt.train_percentage,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=1,
                        checkpoint_callback=True,
                        callbacks=[checkpoint_callback,early_stop_callback],
                        reload_dataloaders_every_epoch=True,
                        num_sanity_val_steps=1,  # Skip Sanity Check
                    )

        
                elif opt.ssl_type == "simclr":

                    augmentaions = get_augmentations(opt)
                    encoder = generate_model(model_depth=opt.model_depth,n_input_channels=1,n_classes=opt.latent_dim,include_fc=False)
                    dm = ParanasalSimCLRDataModule(root_dir=opt.root_dir,dimension=opt.dimension,batch_size=opt.batch_size,trainset=trainset,valset=valset,testset=testset,augmentations=augmentaions)
                    pl_model = PL_MODEL_SIMCLR(model=encoder,opt=opt,expt_name=f"{opt.expt_name}_{opt.seed}_{inner_fold}",result_txt_name=f"{opt.expt_name}")

                    early_stop_callback = EarlyStopping(monitor="val/loss_epoch", min_delta=0.00, patience=20, verbose=True, mode="min")
                    checkpoint_callback = ModelCheckpoint(
                        monitor="val/loss_epoch",
                        dirpath=f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/checkpoint/cv_expts/{opt.expt_name}/",
                        filename=f"model_{opt.seed}_{inner_fold}",
                        mode="min",
                        verbose=True,
                    )

                    trainer = Trainer(
                        max_epochs =opt.epoch,
                        gpus=1,
                        logger=None,
                        limit_train_batches=opt.train_percentage,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=1,
                        checkpoint_callback=True,
                        callbacks=[checkpoint_callback,early_stop_callback],
                        reload_dataloaders_every_epoch=True,
                        num_sanity_val_steps=1,  # Skip Sanity Check
                    )


                
                elif opt.ssl_type == "ours":
                    
                    augmentaions = get_augmentations(opt)
                    encoder = generate_model(model_depth=opt.model_depth,n_input_channels=1,n_classes=opt.latent_dim,include_fc=False)
                    dm = ParanasalContrastiveDataModule(root_dir=opt.root_dir,dimension=opt.dimension,batch_size=opt.batch_size,trainset=trainset,valset=valset,testset=testset,augmentations=augmentaions)
                    callbacks = []
                    if opt.ckpt_path != "None":

                        ckpt_path = opt.ckpt_path + f"/model_{opt.seed}_{inner_fold}.ckpt"

                  
                        pl_model = PL_MODEL(model=encoder,opt=opt,expt_name=f"{opt.expt_name}_{opt.seed}_{inner_fold}",result_txt_name=f"{opt.expt_name}",freeze_enc=True).load_from_checkpoint(
                                        model=encoder,opt=opt,freeze_enc=True,expt_name=f"{opt.expt_name}_{opt.seed}_{inner_fold}",result_txt_name=f"{opt.expt_name}",
                                        checkpoint_path= opt.ckpt_path)
                       

                        checkpoint_callback = ModelCheckpoint(
                            monitor="val/loss_epoch",
                            dirpath=f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/checkpoint/cv_expts/{opt.expt_name}/",
                            filename=f"model_{opt.seed}_{inner_fold}_ft",
                            mode="min",
                            verbose=True,
                        )

                        early_stop_callback = EarlyStopping(monitor="val/loss_epoch", min_delta=0.00, patience=20, verbose=True, mode="min")
                        callbacks = [early_stop_callback,checkpoint_callback]
                    else: 

                        if opt.losses == 1:   

                            pl_model = PL_MODEL(model=encoder,opt=opt,expt_name=f"{opt.expt_name}_{opt.seed}_{inner_fold}",result_txt_name=f"{opt.expt_name}",freeze_enc=False)

                        else: 

                            pl_model = PL_MODEL_2(model=encoder,opt=opt,expt_name=f"{opt.expt_name}_{opt.seed}_{inner_fold}",result_txt_name=f"{opt.expt_name}",freeze_enc=False)

                        early_stop_callback = EarlyStopping(monitor="val/loss_epoch", min_delta=0.00, patience=20, verbose=True, mode="min")
                        checkpoint_callback = ModelCheckpoint(
                            monitor="val/loss_epoch",
                            dirpath=f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/checkpoint/cv_expts/{opt.expt_name}/",
                            filename=f"model_{opt.seed}_{inner_fold}",
                            mode="min",
                            verbose=True,
                        )
                        callbacks = [checkpoint_callback,early_stop_callback]

                    trainer = Trainer(
                        max_epochs =opt.epoch,
                        gpus=1,
                        logger=None,
                        limit_train_batches=opt.train_percentage,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=1,
                        checkpoint_callback=True,
                        callbacks=callbacks,
                        reload_dataloaders_every_epoch=True,
                        num_sanity_val_steps=1,  # Skip Sanity Check
                    )



                trainer.fit(pl_model, datamodule=dm)

                if opt.finetune: trainer.test(ckpt_path="best")
                if opt.ssl_type == "ours" or opt.ssl_type == "simclr" : trainer.test(ckpt_path="best")

            break
    