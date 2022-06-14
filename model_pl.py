import time
from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
from torch import optim

from torch.utils.data import random_split, DataLoader
import monai
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from resnet import generate_model
import torch.nn.functional as F
import numpy as np
from utils import create_dir
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.cm as cm

class PL_MODEL_FT(pl.LightningModule):

    def __init__(self,model,opt,ckpt_path,cls_wt,expt_name,result_txt_name):
        super(PL_MODEL_FT, self).__init__()
        self.model = model

        self.opt = opt
        print(f"Loading pretrained weights {ckpt_path}")
        try:
            self.model.load_state_dict(torch.load(ckpt_path)) 
            print("Loaded pretrained weights")
        except:
            print("Initialised model from scratch")
        self.criterion = torch.nn.CrossEntropyLoss(weight=cls_wt.float()).cuda()
        self.expt_name = expt_name
        self.result_txt_name = result_txt_name
        self.model.fc =  nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2))
        self.softmax = nn.Softmax(dim=1)
        self.pred_all = []
        self.gt_all  = []
        self.pred_conf  = []
    

    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.model.parameters() ,lr=self.opt.lr, amsgrad=False)
        return [optimizer],[]

    def prepare_batch(self, batch):

        return batch['image'][tio.DATA],batch['label']
    
    def infer_batch(self, batch):
        x,y = self.prepare_batch(batch)
        #print("Hello",y)
        y_hat = self.model(x)
        return y_hat,y

    def training_step(self, batch, batch_idx):
        
        #self.counter += 1
        batch_ = batch["train"]
        y_hat,y = self.infer_batch(batch_)
        
        loss = self.criterion(y_hat,y) 
        
        pred_sm = self.softmax(y_hat)
        pred_sm = torch.argmax(pred_sm,dim=1)
        #print(y,pred_sm)
        self.log("train/loss",loss)

        return loss 

    
    def validation_step(self, batch, batch_idx):
        #self.counter += 1

        y_hat,y = self.infer_batch(batch)
        
        loss = self.criterion(y_hat,y) 
        self.log("val/loss",loss, on_step=True,on_epoch=True)

        pred_sm = self.softmax(y_hat)
        pred_sm = torch.argmax(pred_sm,dim=1)

        y  = y.cpu().numpy()
        pred_sm = pred_sm.cpu().numpy()

        accuracy = accuracy_score(y,pred_sm)
        f1_mic =f1_score( y,pred_sm, average='micro')
        f1_mac =f1_score( y,pred_sm, average='macro')
        f1_w =f1_score( y,pred_sm, average='weighted')

        print(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w}")

        return loss.detach() 

    def test_step(self, batch, batch_idx):

        #self.counter += 1
    
        y_hat,y = self.infer_batch(batch)
            
        loss = self.criterion(y_hat,y) 
        self.log("test/loss",loss)

        pred_sm = self.softmax(y_hat)

        conf_abnormal = pred_sm[:,1].cpu().numpy()
        pred_sm = torch.argmax(pred_sm,dim=1)
        pred_sm = pred_sm.cpu().numpy()

        for x in conf_abnormal:
            self.pred_conf.append(x)


        for x in pred_sm:
            self.pred_all.append(x)

        y  = y.cpu().numpy()

        for x in y:
            self.gt_all.append(x)

        return loss.detach()  


    def test_epoch_end(self,output):
        

        accuracy = accuracy_score(self.gt_all,self.pred_all)
        f1_mic =f1_score(self.gt_all,self.pred_all, average='micro')
        f1_mac =f1_score( self.gt_all,self.pred_all, average='macro')
        f1_w =f1_score(self.gt_all,self.pred_all, average='weighted')
 
        aucroc = roc_auc_score(self.gt_all, self.pred_conf)
 
        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}.txt","a+") as f: 

            f.write(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w} AUROC:{aucroc}\n")

        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}_gtvspred.txt","a+") as f: 
            str_pred = ",".join([str(x) for x in self.pred_all])
            str_pred_conf = ",".join([str(x) for x in self.pred_conf])
            str_y = ",".join([str(x) for x in self.gt_all])
            f.write(f"{self.expt_name} GT:{str_y} Pred:{str_pred} Pred_Conf:{str_pred_conf}\n")




class PL_MODEL_FT_2(pl.LightningModule):

    def __init__(self,model,opt,ckpt_path,cls_wt,expt_name,result_txt_name):
        super(PL_MODEL_FT_2, self).__init__()
        self.model = model

        self.opt = opt
        print(f"Loading pretrained weights {ckpt_path}")
        try:
            self.model.load_state_dict(torch.load(ckpt_path)) 
            print("Loaded pretrained weights")
        except:
            print("Initialised model from scratch")
        self.criterion = torch.nn.CrossEntropyLoss(weight=cls_wt.float()).cuda()

        self.projection_head_non_linear_1 = nn.Linear(512, opt.latent_dim)
        self.relu = nn.ReLU()
        self.projection_head_non_linear_2 = nn.Linear(opt.latent_dim, 2)
        self.expt_name = expt_name
        self.result_txt_name = result_txt_name
        self.model.fc =  nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2))
        self.val_counter = 0
        self.softmax = nn.Softmax(dim=1)
        self.pred_all = []
        self.gt_all  = []
        self.pred_conf  = []
    

    def configure_optimizers(self):
        
        optimizer = optim.Adam(list(self.model.parameters())+list(self.projection_head_non_linear_1.parameters()) +list(self.projection_head_non_linear_2.parameters()) ,lr=self.opt.lr, amsgrad=False)
        return [optimizer],[]

    def prepare_batch(self, batch):

        return batch['image'][tio.DATA],batch['label']
    
    def infer_batch(self, batch):
        x,y = self.prepare_batch(batch)

        embeddings = self.model(x)
        y_hat = self.projection_head_non_linear_2(self.relu(self.projection_head_non_linear_1(self.model(x))))

        return y_hat,y,embeddings
        #print("Hello",y)
        

    def training_step(self, batch, batch_idx):
        
        #self.counter += 1
        batch_ = batch["train"]
        y_hat,y,embeddings = self.infer_batch(batch_)
        
        loss = self.criterion(y_hat,y) 
        
        pred_sm = self.softmax(y_hat)
        pred_sm = torch.argmax(pred_sm,dim=1)
        #print(y,pred_sm)
        self.log("train/loss",loss)

        return loss 

    
    def validation_step(self, batch, batch_idx):
        self.val_counter += 1

        y_hat,y,embeddings = self.infer_batch(batch)

        embeddings = F.normalize(embeddings)
        X_embedded = TSNE(n_components=2,init='random').fit_transform(embeddings.cpu().numpy())

        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, 2))
        class_0_embeddings = X_embedded[y.cpu().numpy()==0,:]
        class_1_embeddings = X_embedded[y.cpu().numpy()==1,:]

        ax.scatter(class_0_embeddings[:,0], class_0_embeddings[:,1], color=colors[0],label="Normal")
        ax.scatter(class_1_embeddings[:,0], class_1_embeddings[:,1], color=colors[1],label="Abnormal")

        create_dir(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/')
        plt.savefig(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/{self.val_counter}.png')

        plt.clf()
        
        loss = self.criterion(y_hat,y) 
        self.log("val/loss",loss, on_step=True,on_epoch=True)

        pred_sm = self.softmax(y_hat)
        pred_sm = torch.argmax(pred_sm,dim=1)

        y  = y.cpu().numpy()
        pred_sm = pred_sm.cpu().numpy()

        accuracy = accuracy_score(y,pred_sm)
        f1_mic =f1_score( y,pred_sm, average='micro')
        f1_mac =f1_score( y,pred_sm, average='macro')
        f1_w =f1_score( y,pred_sm, average='weighted')

        print(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w}")

        return loss.detach() 

    def test_step(self, batch, batch_idx):

        #self.counter += 1
    
        y_hat,y,_ = self.infer_batch(batch)
            
        loss = self.criterion(y_hat,y) 
        self.log("test/loss",loss)

        pred_sm = self.softmax(y_hat)

        conf_abnormal = pred_sm[:,1].cpu().numpy()
        pred_sm = torch.argmax(pred_sm,dim=1)
        pred_sm = pred_sm.cpu().numpy()

        for x in conf_abnormal:
            self.pred_conf.append(x)


        for x in pred_sm:
            self.pred_all.append(x)

        y  = y.cpu().numpy()

        for x in y:
            self.gt_all.append(x)

        return loss.detach()  


    def test_epoch_end(self,output):
        

        accuracy = accuracy_score(self.gt_all,self.pred_all)
        f1_mic =f1_score(self.gt_all,self.pred_all, average='micro')
        f1_mac =f1_score( self.gt_all,self.pred_all, average='macro')
        f1_w =f1_score(self.gt_all,self.pred_all, average='weighted')
 
        aucroc = roc_auc_score(self.gt_all, self.pred_conf)
 
        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}.txt","a+") as f: 

            f.write(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w} AUROC:{aucroc}\n")

        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}_gtvspred.txt","a+") as f: 
            str_pred = ",".join([str(x) for x in self.pred_all])
            str_pred_conf = ",".join([str(x) for x in self.pred_conf])
            str_y = ",".join([str(x) for x in self.gt_all])
            f.write(f"{self.expt_name} GT:{str_y} Pred:{str_pred} Pred_Conf:{str_pred_conf}\n")


class PL_MODEL(pl.LightningModule):

    def __init__(self,model,opt,expt_name,result_txt_name ,freeze_enc=False):
        super(PL_MODEL, self).__init__()
        
        self.model = model
        self.opt = opt
        self.expt_name = expt_name
        self.result_txt_name = result_txt_name
  
        self.freeze_enc = freeze_enc

        self.projection_head_non_linear_1 = nn.Linear(512, opt.latent_dim)
        self.relu = nn.ReLU()
        self.projection_head_non_linear_2 = nn.Linear(opt.latent_dim, opt.latent_dim)
        
        self.projection_head_linear_1 = nn.Linear(512, opt.latent_dim)
        self.projection_head_linear_2 = nn.Linear(opt.latent_dim,2)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.counter = 0
        self.training_loss = 0
        self.best_loss = 1e6
        self.pred_all = []
        self.gt_all  = []
        self.softmax  = nn.Softmax()
        self.pred_conf  = []

    def configure_optimizers(self):
            

        optimizer_1 = optim.Adam(list(self.model.parameters())+list(self.projection_head_non_linear_1.parameters()) +list(self.projection_head_non_linear_2.parameters()), lr=self.opt.lr, amsgrad=False)
        optimizer_2 = optim.Adam(list(self.projection_head_linear_1.parameters()) + list(self.projection_head_linear_2.parameters()), lr=self.opt.lr, amsgrad=False)
        return [optimizer_1,optimizer_2],[]


    def info_nce_loss(self, features,neg_features):

        labels = torch.cat([torch.arange(features.shape[0]) for i in range(1)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        neg_mat_1 = neg_features


        positive_matrix   = torch.matmul(features, features.T)
        negative_matrix_1 = torch.matmul(features, neg_mat_1.T)
 
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        positive_matrix = positive_matrix[~mask].view(positive_matrix.shape[0], -1)
        loss =  0

        for i in range(positive_matrix.shape[1]):
            
            feature_map = positive_matrix[:,i:i+1]
            feature_map  = torch.cat((feature_map,negative_matrix_1),1)
            labels = torch.zeros(feature_map.shape[0], dtype=torch.long).cuda()
            loss = loss + self.criterion(feature_map/self.opt.temperature,labels)
        
        loss = loss/(positive_matrix.shape[1])

        return loss


    def prepare_batch(self, batch):
        return batch['image'][tio.DATA],batch['label']
    
    def infer_batch(self, batch,optimizer_idx):
        x,y = self.prepare_batch(batch)

        embeddings = self.model(x)

        if optimizer_idx == 1:

            vector = self.projection_head_linear_2(self.projection_head_linear_1(self.model(x)))

        else:

            vector = self.projection_head_non_linear_2(self.relu(self.projection_head_non_linear_1(self.model(x))))
        
        return vector,y,embeddings

    def training_step(self, batch, batch_idx,optimizer_idx):

        self.counter += 1

          

        class_1_vecs,y_1,embeddings_1  = self.infer_batch(batch["cls_1"],optimizer_idx)
        class_2_vecs,y_2,embeddings_2  = self.infer_batch(batch["cls_2"],optimizer_idx)


        

        if optimizer_idx == 0:

            loss_1 = self.info_nce_loss(F.normalize(class_1_vecs),(F.normalize(class_2_vecs))) 
            loss_2 = self.info_nce_loss(F.normalize(class_2_vecs),(F.normalize(class_1_vecs))) 
            loss = (loss_1 + loss_2)/2
            #loss = loss_2

            embeddings_1 = F.normalize(embeddings_1)
            embeddings_2 = F.normalize(embeddings_2)
            
            embeddings =  torch.cat((embeddings_1,embeddings_2),0)
            y =  torch.cat((y_1,y_2),0)

            X_embedded = TSNE(n_components=2,init='random').fit_transform(embeddings.detach().cpu().numpy())

            fig, ax = plt.subplots()
            colors = cm.rainbow(np.linspace(0, 1, 2))
            class_0_embeddings = X_embedded[y.cpu().numpy()==0,:]
            class_1_embeddings = X_embedded[y.cpu().numpy()==1,:]

            ax.scatter(class_0_embeddings[:,0], class_0_embeddings[:,1], color=colors[0],label="Normal")
            ax.scatter(class_1_embeddings[:,0], class_1_embeddings[:,1], color=colors[1],label="Abnormal")

            create_dir(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/train/')
            plt.savefig(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/train/{self.counter}.png')

            plt.clf()

        else: 

            
            class_vecs=  torch.cat((class_1_vecs,class_2_vecs),0)
            y =  torch.cat((y_1,y_2),0)
            loss = self.criterion(class_vecs,y) 
       
        self.training_loss += loss.item()
        self.log("train/loss",loss, on_step=True,on_epoch=True)

        
        return loss


    def validation_step(self, batch, batch_idx):
        self.counter += 1

        y_hat,y,embeddings = self.infer_batch(batch,1)
        
        
        embeddings = F.normalize(embeddings)
        X_embedded = TSNE(n_components=2,init='random').fit_transform(embeddings.cpu().numpy())

        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, 2))
        class_0_embeddings = X_embedded[y.cpu().numpy()==0,:]
        class_1_embeddings = X_embedded[y.cpu().numpy()==1,:]

        ax.scatter(class_0_embeddings[:,0], class_0_embeddings[:,1], color=colors[0],label="Normal")
        ax.scatter(class_1_embeddings[:,0], class_1_embeddings[:,1], color=colors[1],label="Abnormal")

        create_dir(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/')
        plt.savefig(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/{self.counter}.png')

        plt.clf()


        loss = self.criterion(y_hat,y) 
        self.log("val/loss",loss, on_step=True,on_epoch=True)

        pred_sm = self.softmax(y_hat)
        pred_sm = torch.argmax(pred_sm,dim=1)

        y  = y.cpu().numpy()
        pred_sm = pred_sm.cpu().numpy()

        accuracy = accuracy_score(y,pred_sm)
        f1_mic =f1_score( y,pred_sm, average='micro')
        f1_mac =f1_score( y,pred_sm, average='macro')
        f1_w =f1_score( y,pred_sm, average='weighted')

        print(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w}")

        return loss.detach() 

    def test_step(self, batch, batch_idx):

        #self.counter += 1

        y_hat,y,embeddings_1 = self.infer_batch(batch,1)

        embeddings = F.normalize(embeddings_1)
        X_embedded = TSNE(n_components=2,init='random').fit_transform(embeddings.cpu().numpy())

        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, 2))
        class_0_embeddings = X_embedded[y.cpu().numpy()==0,:]
        class_1_embeddings = X_embedded[y.cpu().numpy()==1,:]

        ax.scatter(class_0_embeddings[:,0], class_0_embeddings[:,1], color=colors[0],label="Normal")
        ax.scatter(class_1_embeddings[:,0], class_1_embeddings[:,1], color=colors[1],label="Abnormal")

        create_dir(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/')
        plt.savefig(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/{self.counter}.png')

        plt.clf()
            
        loss = self.criterion(y_hat,y) 
        self.log("test/loss",loss)

        pred_sm = self.softmax(y_hat)

        conf_abnormal = pred_sm[:,1].cpu().numpy()
        pred_sm = torch.argmax(pred_sm,dim=1)
        pred_sm = pred_sm.cpu().numpy()

        for x in conf_abnormal:
            self.pred_conf.append(x)


        for x in pred_sm:
            self.pred_all.append(x)

        y  = y.cpu().numpy()

        for x in y:
            self.gt_all.append(x)

        return loss.detach()  


    def test_epoch_end(self,output):
        

        accuracy = accuracy_score(self.gt_all,self.pred_all)
        f1_mic =f1_score(self.gt_all,self.pred_all, average='micro')
        f1_mac =f1_score( self.gt_all,self.pred_all, average='macro')
        f1_w =f1_score(self.gt_all,self.pred_all, average='weighted')
        aucroc = roc_auc_score(self.gt_all, self.pred_conf)
 
        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}.txt","a+") as f: 

            f.write(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w} AUROC:{aucroc}\n")


        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}_gtvspred.txt","a+") as f: 


            str_pred = ",".join([str(x) for x in self.pred_all])
            str_pred_conf = ",".join([str(x) for x in self.pred_conf])
            str_y = ",".join([str(x) for x in self.gt_all])
            f.write(f"{self.expt_name} GT:{str_y} Pred:{str_pred} Pred_Conf:{str_pred_conf}\n")
   


#Our proposed model that uses contrastive loss and cross entropy loss
class PL_MODEL_2(pl.LightningModule):

    def __init__(self,model,opt,expt_name,result_txt_name ,freeze_enc=False):
        super(PL_MODEL_2, self).__init__()
        
        self.model = model
        self.opt = opt
        self.expt_name = expt_name
        self.result_txt_name = result_txt_name
  
        self.freeze_enc = freeze_enc

        self.projection_head_non_linear_1 = nn.Linear(512, opt.latent_dim)
        self.relu = nn.ReLU()
        self.projection_head_non_linear_2 = nn.Linear(opt.latent_dim, opt.latent_dim)
        
        self.projection_head_linear_1 = nn.Linear(512, opt.latent_dim)
        self.projection_head_linear_2 = nn.Linear(opt.latent_dim,2)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.counter = 0
        self.val_counter = 0
        self.training_loss = 0
        self.best_loss = 1e6
        self.pred_all = []
        self.gt_all  = []
        self.softmax  = nn.Softmax()
        self.pred_conf  = []

    def configure_optimizers(self):
            

        optimizer = optim.Adam(list(self.model.parameters())+list(self.projection_head_non_linear_1.parameters()) +list(self.projection_head_non_linear_2.parameters()) + list(self.projection_head_linear_1.parameters()) + list(self.projection_head_linear_2.parameters()), lr=self.opt.lr, amsgrad=False)
        return [optimizer],[]


    def info_nce_loss(self, features,neg_features):

        labels = torch.cat([torch.arange(features.shape[0]) for i in range(1)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        neg_mat_1 = neg_features

    
        positive_matrix   = torch.matmul(features, features.T)
        negative_matrix_1 = torch.matmul(features, neg_mat_1.T)
 
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        positive_matrix = positive_matrix[~mask].view(positive_matrix.shape[0], -1)
        loss =  0

        for i in range(positive_matrix.shape[1]):

            feature_map = positive_matrix[:,i:i+1]
            feature_map  = torch.cat((feature_map,negative_matrix_1),1)
            labels = torch.zeros(feature_map.shape[0], dtype=torch.long).cuda()
            loss = loss + self.criterion(feature_map/self.opt.temperature,labels)
        
        loss = loss/(positive_matrix.shape[1])

        return loss


    def prepare_batch(self, batch):
        return batch['image'][tio.DATA],batch['label']
    
    def infer_batch(self, batch):
        x,y = self.prepare_batch(batch)

        embeddings = self.model(x)

        

        vector_linear = self.projection_head_linear_2(self.projection_head_linear_1(self.model(x)))

        vector_non_linear = self.projection_head_non_linear_2(self.relu(self.projection_head_non_linear_1(self.model(x))))
        
        return vector_linear,vector_non_linear,y,embeddings

    def training_step(self, batch, batch_idx):

        self.counter += 1

          

        class_1_vecs_linear,class_1_vecs_nonlinear,y_1,embeddings_1  = self.infer_batch(batch["cls_1"])
        class_2_vecs_linear,class_2_vecs_nonlinear,y_2,embeddings_2  = self.infer_batch(batch["cls_2"])

        
        loss_1 = self.info_nce_loss(F.normalize(class_1_vecs_nonlinear),(F.normalize(class_2_vecs_nonlinear))) 
        loss_2 = self.info_nce_loss(F.normalize(class_2_vecs_nonlinear),(F.normalize(class_1_vecs_nonlinear))) 
        
        embeddings_1 = F.normalize(embeddings_1)
        embeddings_2 = F.normalize(embeddings_2)
        
        embeddings =  torch.cat((embeddings_1,embeddings_2),0)
        y =  torch.cat((y_1,y_2),0)

        X_embedded = TSNE(n_components=2,init='random').fit_transform(embeddings.detach().cpu().numpy())

        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, 2))
        class_0_embeddings = X_embedded[y.cpu().numpy()==0,:]
        class_1_embeddings = X_embedded[y.cpu().numpy()==1,:]

        ax.scatter(class_0_embeddings[:,0], class_0_embeddings[:,1], color=colors[0],label="Normal")
        ax.scatter(class_1_embeddings[:,0], class_1_embeddings[:,1], color=colors[1],label="Abnormal")

        create_dir(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/train/')
        plt.savefig(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/train/{self.counter}.png')

        plt.clf()

        contrastive_loss = (loss_1 + loss_2)/2
        #loss = loss_2
        print("Contrastive",contrastive_loss.item())

        class_vecs=  torch.cat((class_1_vecs_linear,class_2_vecs_linear),0)
        y =  torch.cat((y_1,y_2),0)

        sup_loss = self.criterion(class_vecs,y) 
        print("Supervised loss",sup_loss.item())
        loss = contrastive_loss + sup_loss
        
       
        self.training_loss += loss.item()
        self.log("train/loss",loss, on_step=True,on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        self.val_counter += 1
        y_hat,class_1_vecs_nonlinear,y,embeddings = self.infer_batch(batch)
        embeddings = F.normalize(embeddings)
        X_embedded = TSNE(n_components=2,init='random').fit_transform(embeddings.cpu().numpy())

        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, 2))
        class_0_embeddings = X_embedded[y.cpu().numpy()==0,:]
        class_1_embeddings = X_embedded[y.cpu().numpy()==1,:]

        ax.scatter(class_0_embeddings[:,0], class_0_embeddings[:,1], color=colors[0],label="Normal")
        ax.scatter(class_1_embeddings[:,0], class_1_embeddings[:,1], color=colors[1],label="Abnormal")

        create_dir(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/')
        plt.savefig(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/{self.val_counter}.png')

        plt.clf()



        loss = self.criterion(y_hat,y) 
        self.log("val/loss",loss, on_step=True,on_epoch=True)

        pred_sm = self.softmax(y_hat)
        pred_sm = torch.argmax(pred_sm,dim=1)

        y  = y.cpu().numpy()
        pred_sm = pred_sm.cpu().numpy()

        accuracy = accuracy_score(y,pred_sm)
        f1_mic =f1_score( y,pred_sm, average='micro')
        f1_mac =f1_score( y,pred_sm, average='macro')
        f1_w =f1_score( y,pred_sm, average='weighted')

        print(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w}")

        return loss.detach() 

    def test_step(self, batch, batch_idx):

        #self.counter += 1

        y_hat,class_1_vecs_nonlinear,y,embeddings = self.infer_batch(batch)
        
        X_embedded = TSNE(n_components=2,init='random').fit_transform(embeddings.cpu().numpy())

        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, 2))
        class_0_embeddings = X_embedded[y.cpu().numpy()==0,:]
        class_1_embeddings = X_embedded[y.cpu().numpy()==1,:]

        ax.scatter(class_0_embeddings[:,0], class_0_embeddings[:,1], color=colors[0],label="Normal")
        ax.scatter(class_1_embeddings[:,0], class_1_embeddings[:,1], color=colors[1],label="Abnormal")

        create_dir(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/test/')
        plt.savefig(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/test/test.png')

        plt.clf()
        loss = self.criterion(y_hat,y) 
        self.log("test/loss",loss)

        pred_sm = self.softmax(y_hat)

        conf_abnormal = pred_sm[:,1].cpu().numpy()
        pred_sm = torch.argmax(pred_sm,dim=1)
        pred_sm = pred_sm.cpu().numpy()

        for x in conf_abnormal:
            self.pred_conf.append(x)


        for x in pred_sm:
            self.pred_all.append(x)

        y  = y.cpu().numpy()

        for x in y:
            self.gt_all.append(x)

        return loss.detach()  


    def test_epoch_end(self,output):
        

        accuracy = accuracy_score(self.gt_all,self.pred_all)
        f1_mic =f1_score(self.gt_all,self.pred_all, average='micro')
        f1_mac =f1_score( self.gt_all,self.pred_all, average='macro')
        f1_w =f1_score(self.gt_all,self.pred_all, average='weighted')
        aucroc = roc_auc_score(self.gt_all, self.pred_conf)
 
        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}.txt","a+") as f: 

            f.write(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w} AUROC:{aucroc}\n")


        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}_gtvspred.txt","a+") as f: 
            str_pred = ",".join([str(x) for x in self.pred_all])
            str_pred_conf = ",".join([str(x) for x in self.pred_conf])
            str_y = ",".join([str(x) for x in self.gt_all])
            f.write(f"{self.expt_name} GT:{str_y} Pred:{str_pred} Pred_Conf:{str_pred_conf}\n")
   
  

class PL_MODEL_3(pl.LightningModule):

    def __init__(self,model,opt,expt_name,result_txt_name ,freeze_enc=False):
        super(PL_MODEL_3, self).__init__()
        
        self.model = model
        self.opt = opt
        self.expt_name = expt_name
        self.result_txt_name = result_txt_name
  
        self.freeze_enc = freeze_enc

        self.projection_head_non_linear_1 = nn.Linear(512, opt.latent_dim)
        self.relu = nn.ReLU()
        self.projection_head_non_linear_2 = nn.Linear(opt.latent_dim, opt.latent_dim)
        
        self.projection_head_linear_1 = nn.Linear(512, opt.latent_dim)
        self.projection_head_linear_2 = nn.Linear(opt.latent_dim,2)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.counter = 0
        self.training_loss = 0
        self.best_loss = 1e6
        self.pred_all = []
        self.gt_all  = []
        self.softmax  = nn.Softmax()
        self.pred_conf  = []

    def configure_optimizers(self):
            

        optimizer = optim.Adam(list(self.model.parameters())+list(self.projection_head_non_linear_1.parameters()) +list(self.projection_head_non_linear_2.parameters()) + list(self.projection_head_linear_1.parameters()) + list(self.projection_head_linear_2.parameters()), lr=self.opt.lr, amsgrad=False)
        return [optimizer],[]


    def info_nce_loss(self, features,neg_features):

        labels = torch.cat([torch.arange(features.shape[0]) for i in range(1)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        neg_mat_1 = neg_features

    
        positive_matrix   = torch.matmul(features, features.T)
        negative_matrix_1 = torch.matmul(features, neg_mat_1.T)
 
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        positive_matrix = positive_matrix[~mask].view(positive_matrix.shape[0], -1)
        loss =  0

        for i in range(positive_matrix.shape[1]):

            feature_map = positive_matrix[:,i:i+1]
            feature_map  = torch.cat((feature_map,negative_matrix_1),1)
            labels = torch.zeros(feature_map.shape[0], dtype=torch.long).cuda()
            loss = loss + self.criterion(feature_map/self.opt.temperature,labels)
        
        loss = loss/(positive_matrix.shape[1])

        return loss


    def prepare_batch(self, batch):

        return batch['image'][tio.DATA],batch['label']
    
    def infer_batch(self, batch):
        x,y = self.prepare_batch(batch)

        embeddings = self.model(x)

        vector_linear = self.projection_head_linear_2(self.projection_head_linear_1(self.model(x)))

        vector_non_linear = self.projection_head_non_linear_2(self.relu(self.projection_head_non_linear_1(self.model(x))))
        
        return vector_linear,vector_non_linear,y,embeddings

    def training_step(self, batch, batch_idx):

        self.counter += 1

          

        class_1_vecs_linear,class_1_vecs_nonlinear,y_1,embeddings_1  = self.infer_batch(batch["cls_1"])
        class_2_vecs_linear,class_2_vecs_nonlinear,y_2,embeddings_2  = self.infer_batch(batch["cls_2"])
        
        loss_1 = self.info_nce_loss(F.normalize(class_1_vecs_nonlinear),(F.normalize(class_2_vecs_nonlinear))) 
        loss_2 = self.info_nce_loss(F.normalize(class_2_vecs_nonlinear),(F.normalize(class_1_vecs_nonlinear))) 

        contrastive_loss = (loss_1 + loss_2)/2
        #loss = loss_2
        print("Contrastive",contrastive_loss.item())

        class_vecs=  torch.cat((class_1_vecs_linear,class_2_vecs_linear),0)
        y =  torch.cat((y_1,y_2),0)
        
        sup_loss = self.criterion(class_vecs,y) 
        print("Supervised loss",sup_loss.item())
        loss = contrastive_loss + sup_loss
        
       
        self.training_loss += loss.item()
        self.log("train/loss",loss, on_step=True,on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        #self.counter += 1

        y_hat,class_1_vecs_nonlinear,y,embeddings = self.infer_batch(batch)
        
        loss = self.criterion(y_hat,y) 
        self.log("val/loss",loss, on_step=True,on_epoch=True)

        pred_sm = self.softmax(y_hat)
        pred_sm = torch.argmax(pred_sm,dim=1)

        y  = y.cpu().numpy()
        pred_sm = pred_sm.cpu().numpy()

        accuracy = accuracy_score(y,pred_sm)
        f1_mic =f1_score( y,pred_sm, average='micro')
        f1_mac =f1_score( y,pred_sm, average='macro')
        f1_w =f1_score( y,pred_sm, average='weighted')

        print(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w}")

        return loss.detach() 

    def test_step(self, batch, batch_idx):

        #self.counter += 1

        y_hat,class_1_vecs_nonlinear,y,embeddings = self.infer_batch(batch)
            
        loss = self.criterion(y_hat,y) 
        self.log("test/loss",loss)

        pred_sm = self.softmax(y_hat)

        conf_abnormal = pred_sm[:,1].cpu().numpy()
        pred_sm = torch.argmax(pred_sm,dim=1)
        pred_sm = pred_sm.cpu().numpy()

        for x in conf_abnormal:
            self.pred_conf.append(x)


        for x in pred_sm:
            self.pred_all.append(x)

        y  = y.cpu().numpy()

        for x in y:
            self.gt_all.append(x)

        return loss.detach()  


    def test_epoch_end(self,output):
        

        accuracy = accuracy_score(self.gt_all,self.pred_all)
        f1_mic =f1_score(self.gt_all,self.pred_all, average='micro')
        f1_mac =f1_score( self.gt_all,self.pred_all, average='macro')
        f1_w =f1_score(self.gt_all,self.pred_all, average='weighted')
        aucroc = roc_auc_score(self.gt_all, self.pred_conf)
 
        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}.txt","a+") as f: 

            f.write(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w} AUROC:{aucroc}\n")


        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}_gtvspred.txt","a+") as f: 
            str_pred = ",".join([str(x) for x in self.pred_all])
            str_pred_conf = ",".join([str(x) for x in self.pred_conf])
            str_y = ",".join([str(x) for x in self.gt_all])
            f.write(f"{self.expt_name} GT:{str_y} Pred:{str_pred} Pred_Conf:{str_pred_conf}\n")
   
  

class PL_MODEL_TEST(pl.LightningModule):

    def __init__(self,model,opt):
        super(PL_MODEL_TEST, self).__init__()
        
        self.model = model
        self.opt = opt
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(opt.latent_dim, opt.n_classes))
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.counter = 0
        self.training_loss = 0
        self.best_loss = 1e6
        self.ds_to_save = "smax_0std"

        self.feature_array  = None


    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.fc.parameters(), lr=self.opt.lr, amsgrad=False)
        return [optimizer],[]


    def get_similarity_score(self, features,neg_features):

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


    def prepare_batch(self, batch):

        return batch['image'][tio.DATA]
    
    def infer_batch(self, batch):
        x = self.prepare_batch(batch)
        y_hat = self.model(x)
        return y_hat

    def test_step(self, batch, batch_idx):

        self.counter += 1

        features = F.normalize(self.infer_batch(batch),dim=1)

        if self.feature_array == None: 

            self.feature_array = features

        else: 

            self.feature_array = torch.cat((self.feature_array,features),0)


        return self.feature_array

    
    def test_epoch_end(self,output):

        np.save(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/vectors/{self.ds_to_save}.npy",self.feature_array.cpu().numpy())

    


class PL_MODEL_SIMCLR(pl.LightningModule):

    def __init__(self,model,opt,expt_name,result_txt_name ):
        super(PL_MODEL_SIMCLR, self).__init__()
        
        self.model = model
        self.opt = opt

        self.expt_name = expt_name
        self.result_txt_name = result_txt_name
        
        self.projection_head_non_linear_1 = nn.Linear(512, opt.latent_dim)
        self.relu = nn.ReLU()
        self.projection_head_non_linear_2 = nn.Linear(opt.latent_dim, opt.latent_dim)
        
        self.projection_head_linear_1 = nn.Linear(512, opt.latent_dim)
        self.projection_head_linear_2 = nn.Linear(opt.latent_dim,2)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.counter = 0
        self.val_counter = 0
        self.training_loss = 0
        self.best_loss = 1e6
        self.pred_all = []
        self.gt_all  = []
        self.pred_conf  = []

    def configure_optimizers(self):
        
        optimizer_1 = optim.Adam(list(self.model.parameters())+list(self.projection_head_non_linear_1.parameters()) +list(self.projection_head_non_linear_2.parameters()), lr=self.opt.lr, amsgrad=False)
        optimizer_2 = optim.Adam(list(self.projection_head_linear_1.parameters()) + list(self.projection_head_linear_2.parameters()), lr=self.opt.lr, amsgrad=False)
        return [optimizer_1,optimizer_2],[]

    

    def info_nce_loss_2(self, features_1,features_2,neg_features_1,neg_features_2):

        labels = torch.cat([torch.arange(features_1.shape[0]) for i in range(1)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        neg_mat_1 = neg_features_1
        neg_mat_2 = neg_features_2


        positive_matrix   = torch.matmul(features_1, features_2.T)
        negative_matrix_1 = torch.matmul(features_1, neg_mat_1.T)
        negative_matrix_2 = torch.matmul(features_1, neg_mat_2.T)

        #negative_matrix_3 = torch.matmul(features, neg_mat_3.T)
        #negative_matrix_4 = torch.matmul(features, neg_mat_4.T)
        #negative_matrix_5 = torch.matmul(features, neg_mat_5.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)

        
        #print(positive_matrix)
        positive_matrix = positive_matrix[mask].view(positive_matrix.shape[0], -1)
        loss =  0
        #print(positive_matrix)
        for i in range(positive_matrix.shape[1]):

            feature_map = positive_matrix[:,i:i+1]
            feature_map  = torch.cat((feature_map,negative_matrix_1,negative_matrix_2),1)
            labels = torch.zeros(feature_map.shape[0], dtype=torch.long).cuda()
            loss = loss + self.criterion(feature_map/self.opt.temperature,labels)
        
        loss = loss/(positive_matrix.shape[1])

        return loss

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA]
        
    def prepare_batch(self, batch):
        return batch['image'][tio.DATA],batch['label']
    
    def infer_batch(self, batch,optimizer_idx):
        x,y = self.prepare_batch(batch)

        embeddings = self.model(x)

        if optimizer_idx == 1:

            vector = self.projection_head_linear_2(self.projection_head_linear_1(self.model(x)))

        else:

            vector = self.projection_head_non_linear_2(self.relu(self.projection_head_non_linear_1(self.model(x))))
        
        return vector,y,embeddings

    def training_step(self, batch, batch_idx,optimizer_idx):

        self.counter += 1


        class_1_vecs_v1,y_1_v1,embeddings_1_v1  = self.infer_batch(batch["cls_1_v1"],optimizer_idx)
        class_2_vecs_v1,y_2_v1,embeddings_2_v1  = self.infer_batch(batch["cls_2_v1"],optimizer_idx)

        class_1_vecs_v2,y_1_v2,embeddings_1_v2  = self.infer_batch(batch["cls_1_v2"],optimizer_idx)
        class_2_vecs_v2,y_2_v2,embeddings_2_v2  = self.infer_batch(batch["cls_2_v2"],optimizer_idx)


        if optimizer_idx == 0:
            
            
            loss_1 = self.info_nce_loss_2(F.normalize(class_1_vecs_v1),F.normalize(class_1_vecs_v2),F.normalize(class_2_vecs_v1),F.normalize(class_2_vecs_v2)) 
            loss_2 = self.info_nce_loss_2(F.normalize(class_1_vecs_v2),F.normalize(class_1_vecs_v1),F.normalize(class_2_vecs_v1),F.normalize(class_2_vecs_v2)) 
            loss_3 = self.info_nce_loss_2(F.normalize(class_2_vecs_v1),F.normalize(class_2_vecs_v2),F.normalize(class_1_vecs_v1),F.normalize(class_1_vecs_v2)) 
            loss_4 = self.info_nce_loss_2(F.normalize(class_2_vecs_v2),F.normalize(class_2_vecs_v1),F.normalize(class_1_vecs_v1),F.normalize(class_1_vecs_v1)) 
            
            #print(loss_1.item(),loss_2.item(),loss_3.item(),loss_4.item())
            #print(self.result_txt_name)
            loss = (loss_1 + loss_2 + loss_3 + loss_4)/4
            #loss = loss_2

        else: 

            
            class_vecs=  torch.cat((class_1_vecs_v1,class_2_vecs_v1),0)
            y =  torch.cat((y_1_v1,y_2_v1),0)
            loss = self.criterion(class_vecs,y) 

            class_vecs=  torch.cat((class_1_vecs_v2,class_2_vecs_v2),0)
            y =  torch.cat((y_1_v2,y_2_v2),0)
            loss += self.criterion(class_vecs,y) 

       
        self.training_loss += loss.item()
        self.log("train/loss",loss, on_step=True,on_epoch=True)

        
        return loss


    def validation_step(self, batch, batch_idx):
        #self.counter += 1
        self.val_counter += 1
        y_hat,y,embeddings = self.infer_batch(batch,1)

        embeddings = F.normalize(embeddings)
        X_embedded = TSNE(n_components=2,init='random').fit_transform(embeddings.cpu().numpy())

        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, 2))
        class_0_embeddings = X_embedded[y.cpu().numpy()==0,:]
        class_1_embeddings = X_embedded[y.cpu().numpy()==1,:]

        ax.scatter(class_0_embeddings[:,0], class_0_embeddings[:,1], color=colors[0],label="Normal")
        ax.scatter(class_1_embeddings[:,0], class_1_embeddings[:,1], color=colors[1],label="Abnormal")

        create_dir(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/')
        plt.savefig(f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results/figures/{self.expt_name}/val/{self.val_counter}.png')

        plt.clf()
        plt.close()
        
        loss = self.criterion(y_hat,y) 
        self.log("val/loss",loss, on_step=True,on_epoch=True)

        pred_sm = self.softmax(y_hat)
        pred_sm = torch.argmax(pred_sm,dim=1)

        y  = y.cpu().numpy()
        pred_sm = pred_sm.cpu().numpy()

        accuracy = accuracy_score(y,pred_sm)
        f1_mic =f1_score( y,pred_sm, average='micro')
        f1_mac =f1_score( y,pred_sm, average='macro')
        f1_w =f1_score( y,pred_sm, average='weighted')

        print(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w}")

        return loss.detach() 

    def test_step(self, batch, batch_idx):

        #self.counter += 1

        y_hat,y,embeddings_1 = self.infer_batch(batch,1)
            
        loss = self.criterion(y_hat,y) 
        self.log("test/loss",loss)

        pred_sm = self.softmax(y_hat)

        conf_abnormal = pred_sm[:,1].cpu().numpy()
        pred_sm = torch.argmax(pred_sm,dim=1)
        pred_sm = pred_sm.cpu().numpy()

        for x in conf_abnormal:
            self.pred_conf.append(x)


        for x in pred_sm:
            self.pred_all.append(x)

        y  = y.cpu().numpy()

        for x in y:
            self.gt_all.append(x)

        return loss.detach()  


    def test_epoch_end(self,output):
        

        accuracy = accuracy_score(self.gt_all,self.pred_all)
        f1_mic =f1_score(self.gt_all,self.pred_all, average='micro')
        f1_mac =f1_score( self.gt_all,self.pred_all, average='macro')
        f1_w =f1_score(self.gt_all,self.pred_all, average='weighted')
        aucroc = roc_auc_score(self.gt_all, self.pred_conf)
 
        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}.txt","a+") as f: 

            f.write(f"{self.expt_name} Acc:{accuracy} F1_mic:{f1_mic} F1_mac:{f1_mac} F1_weighted:{f1_w} AUROC:{aucroc}\n")


        with open("//media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet2/results" + f"/{self.result_txt_name}_gtvspred.txt","a+") as f: 
            str_pred = ",".join([str(x) for x in self.pred_all])
            str_pred_conf = ",".join([str(x) for x in self.pred_conf])
            str_y = ",".join([str(x) for x in self.gt_all])
            f.write(f"{self.expt_name} GT:{str_y} Pred:{str_pred} Pred_Conf:{str_pred_conf}\n")
   
  


class PL_MODEL_COMPARE(pl.LightningModule):

    def __init__(self,model,opt,decoder = None):
        super(PL_MODEL_COMPARE, self).__init__()
        
        self.model = model
        self.decoder = decoder
        self.opt = opt
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(opt.latent_dim, opt.n_classes))
        if decoder == None: 
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = torch.nn.MSELoss().cuda()
        self.counter = 0
        self.training_loss = 0
        self.best_loss = 1e6

    def configure_optimizers(self):
        
        optimizer = optim.Adam(list(self.model.parameters())+list(self.fc.parameters()), lr=self.opt.lr, amsgrad=False)
        return [optimizer],[]


    def prepare_batch(self, batch):

        return batch['image'][tio.DATA],batch['label']
    
    def infer_batch(self, batch):

        x,y = self.prepare_batch(batch)
        if self.decoder == None:
            y_hat = self.fc(self.model(x))
        else:
            y = x
            y_hat = self.decoder(self.model(x),None)
        return y_hat,y

    def training_step(self, batch, batch_idx):

        self.counter += 1
        batch_ = batch["smax"]

        y_hat,y = self.infer_batch(batch_)
        
        
        loss = self.criterion(y_hat,y) 

        #loss = (loss1 + loss2 + loss3 + loss4)/4 
        self.training_loss += loss.item()
        self.log("train/loss",loss)

        return loss

    def training_epoch_end(self,output):

        avg_training_loss = self.training_loss/self.counter 

        if self.best_loss > avg_training_loss: 
        
            print("SAVING PYTORCH MODEL")
            self.best_loss = avg_training_loss
            torch.save(self.model.state_dict(),f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/checkpoint/{self.opt.expt_name}/model_enc.ckpt")

        self.training_loss = 0 
        self.counter = 0 



"""
mod = generate_model(model_depth=18,n_input_channels=1,n_classes=128) 

print(mod)
x = torch.rand(1,1,32,32,32)
x = mod(x)
print(x.shape)

"""