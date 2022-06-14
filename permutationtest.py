import pickle
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
#import xlsxwriter
import matplotlib.pyplot as plt
import h5py
import os
from glob import glob
import scipy
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,f1_score,average_precision_score,auc,accuracy_score
from mlxtend.evaluate import permutation_test
import bootstrap
import sklearn
import importlib
from tqdm import tqdm
import argparse
# python read_cv_results_single_last_test_stack_prediction.py  Image Patch_ZVI_3DCNN_small_kernel_no_stride_bal1_2_compr1_fcmcnn2_222 CGC.Patch_ConvGRU_fm12_k7_bal1_0_bi_fmcnn1

def spec_func_threshed(predictions_tresh,gt):
            # Sens/Spec/Acc for different thresholds
            # Confusion matrix
            conf = confusion_matrix(gt,predictions_tresh)

            #print("Conf",conf)
            # Sensitivity / Specificity
            FP = conf.sum(axis=0) - np.diag(conf)
            FN = conf.sum(axis=1) - np.diag(conf)
            TP = np.diag(conf)
            TN = conf.sum() - (FP + FN + TP)
            # Sensitivity
            specificity = TN/(TN+FP)
            #print(specificity)
            return specificity[1]
def acc_func_threshed(predictions_tresh,gt):
    return accuracy_score(gt,predictions_tresh)
def sens_func_threshed(predictions_tresh,gt):
            # Sens/Spec/Acc for different thresholds
            # Confusion matrix
            conf = confusion_matrix(gt,predictions_tresh)

            #print("Conf",conf)
            # Sensitivity / Specificity
            FP = conf.sum(axis=0) - np.diag(conf)
            FN = conf.sum(axis=1) - np.diag(conf)
            TP = np.diag(conf)
            TN = conf.sum() - (FP + FN + TP)
            # Sensitivity
            Sensitivity = TP/(TP+FN)
            #print(specificity)
            return Sensitivity[1]
def f1_func_threshed(predictions_tresh,gt):
            # Sens/Spec/Acc for different thresholds
            # Confusion matrix
            f1 = f1_score(gt,predictions_tresh,average='weighted') 
            return f1
def AUC_func_threshed(predictions_tresh,gt):
            # Sens/Spec/Acc for different thresholds
            # Confusion matrix
            fpr, tpr, ts = roc_curve(gt,predictions_tresh)
            fpr_100pSens = fpr[tpr.argmax()]
            
            AUC = roc_auc_score(gt,predictions_tresh)
            #return 1 - fpr_100pSens
            return AUC
def AUPRC_func_threshed(predictions_tresh,gt):
            # Sens/Spec/Acc for different thresholds
            # Confusion matrix
            AUPRC = average_precision_score(gt,predictions_tresh)
            return AUPRC
def get_confi_intervals(pred,tar,metric):
    # tar = tar[:, 1].astype(int)
    #pred = pred[:, 1]

    if metric == 'AUC':
        #tar = tar[:, 1].astype(int)
        # pred = pred[:, 1]
        cis_sens = bootstrap.ci((pred, tar), lambda x, y: AUC_func_threshed(x,y), n_samples=1000 )
        print('AUC - Konf Intervall', cis_sens)

    elif metric =='F1':
        #tar = np.argmax(tar,1)
        #pred = np.argmax(pred,1)
        cis_sens = bootstrap.ci((pred, tar), lambda x, y: f1_func_threshed(x,y), n_samples=1000 )
        print('F1 - Konf Intervall', cis_sens)

    elif metric == 'Sens':
        #tar = np.argmax(tar,1)
        #pred = np.argmax(pred,1)
        cis_sens = bootstrap.ci((pred, tar), lambda x, y: sens_func_threshed(x,y), n_samples=1000 )
        print('Sens - Konf Intervall', cis_sens)

    elif metric == 'Spec':
        #tar = np.argmax(tar,1)
        #pred = np.argmax(pred,1)
        cis_sens = bootstrap.ci((pred, tar), lambda x, y: spec_func_threshed(x,y), n_samples=1000 )
        print('Spec - Konf Intervall', cis_sens)
    elif metric == 'ACC':
        #tar = np.argmax(tar,1)
        #pred = np.argmax(pred,1)
        cis_sens = bootstrap.ci((pred, tar), lambda x, y: acc_func_threshed(x,y), n_samples=1000 )
        print('ACC - Konf Intervall', cis_sens)
    return cis_sens 



ours = []
baseline = []
ground_truth = []
with open("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/nested_cv_expts/pretrained_ours_3_gtvspred.txt","r") as f:

    lines = f.readlines()
    lines = sorted(lines)

    for line in lines: 

        array = line.split()
        gt = array[1].split(":")[1].split(",") 
        gt = [int(x) for x in gt]

        ground_truth.append(gt)

        pred = array[2].split(":")[1].split(",") 
        pred = [int(x) for x in pred]
        ours.append(pred)



ours = np.array(ours).flatten()
ground_truth = np.array(ground_truth).flatten()

with open("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/ContLearnNet/results/nested_cv_expts/simclr_gtvspred.txt","r") as f:

    lines = f.readlines()
    lines = sorted(lines)

    for line in lines: 
        array = line.split()
    
        pred = array[2].split(":")[1].split(",") 
        pred = [int(x) for x in pred]
        baseline.append(pred)


baseline = np.array(baseline).flatten()

print(baseline.shape)

met = ours # proposed Method



# method, baseline, ground_truth = sklearn.utils.shuffle(method, baseline, ground_truth)
p_value = permutation_test(met, baseline,
                        method='approximate',
                        func = lambda x, y: np.abs(f1_func_threshed(x,ground_truth)-f1_func_threshed(y,ground_truth)),
                        num_rounds=10000)

p_value = permutation_test(met, baseline,
                         method='approximate',
                         func = lambda x, y: np.abs(acc_func_threshed(x,ground_truth)-acc_func_threshed(y,ground_truth)),
                         num_rounds=10000)
                         
# p_value = permutation_test(met, baseline,
#                         method='approximate',
#                         func = lambda x, y: np.abs(sens_func_threshed(x,ground_truth)-sens_func_threshed(y,ground_truth)),
#                         num_rounds=10000)

# p_value = permutation_test(met, baseline,
#                         method='approximate',
#                         func = lambda x, y: np.abs(spec_func_threshed(x,ground_truth)-spec_func_threshed(y,ground_truth)),
#                         num_rounds=10000)


    

"""
ACC0 = AUC_func_threshed(baseline,ground_truth)
ACC1 = AUC_func_threshed(met,ground_truth)
print('########################################################################')
print('Metric of Baseline VAE: {}'.format(ACC0)) 
ci = get_confi_intervals(baseline,ground_truth,'AUC')
print('Metric of proposed Method: {}'.format(ACC1))
ci = get_confi_intervals(met,ground_truth,'AUC')
print('p Value of comparison AUC-score', p_value)


AUC0 = roc_auc_score(targets_comb.astype(int),comb0[0])
AUC1 = roc_auc_score(targets_comb.astype(int),comb1[0])



plt.title('Receiver Operating Characteristic')
plt.plot(fpr0, tpr0, 'r', label = 'AUC Baseline = %0.2f' % AUC0)
plt.plot(fpr1, tpr1, 'b', label = 'AUC Method = %0.2f' % AUC1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#plt.show()

# ground_truth = np.argmax(Targets_all[0],1)
# pred3D = np.array(Pred_all_trash[0])
# pred2DMS = np.array(Pred_all_trash[1])

# p_value = permutation_test(pred3D, pred2DMS,
#                            method='approximate',
#                            func = lambda x, y: np.abs(f1_func_threshed(x,ground_truth)-f1_func_threshed(y,ground_truth)),
#                            num_rounds=10000)

# print('p Value of comparison f1-score', p_value)

# p_value = permutation_test(pred3D, pred2DMS,
#                            method='approximate',
#                            func = lambda x, y: np.abs(spec_func_threshed(x,ground_truth)-spec_func_threshed(y,ground_truth)),
#                            num_rounds=10000)

# print('p Value of comparison Sens-score', p_value)


# p_value = permutation_test(pred3D, pred2DMS,
#                            method='approximate',
#                            func = lambda x, y: np.abs(spec_func_threshed(x,ground_truth)-spec_func_threshed(y,ground_truth)),
#                            num_rounds=10000)

# print('p Value of comparison Spec-score', p_value)

"""





