import os
import time

import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

# Dictionary to store stats
def get_res_dict():
    res = {'acc': None,
           'ddp': None,
           'ppv': None,
           'fpr': None,
           'fnr': None,
           'tn_s0': None,
           'tn_s1': None,
           'fp_s0': None,
           'fp_s1': None,
           'fn_s0': None,
           'fn_s1': None,
           'tp_s0': None,
           'tp_s1': None
           }
    return res

def save_plot(epoch, loss_values, config, file):
    plt.figure()
    plt.plot(np.array(loss_values), 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if not os.path.isdir(config['plots_file']):
        os.mkdir(config['plots_file'])
    plt.savefig(config['plots_file'] + '/' + file + str(epoch+1) + '.png')


def save_avg_plot(epoch, loss_values, config, file):
    plt.figure()
    b = []
    for i in range(len(loss_values)):
        if i < 10:
            b.append(loss_values[i])
        else:
            b.append(np.average([loss_values[j] for j in range(i-10,i)]))
    plt.plot(np.array(b), 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if not os.path.isdir(config['plots_file']):
        os.mkdir(config['plots_file'])
    plt.savefig(config['plots_file'] + '/' + file + str(epoch+1) + '.png')


def save_everything(epoch, net, optimizer, train_acc, val_acc, best, config, file):
    # Save checkpoint.
    state = {
        'epoch': epoch,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best': best
    }
    if not os.path.isdir(config['file_name']):
        os.mkdir(config['file_name'])
    torch.save(state, config['file_name'] + '/' + file + str(epoch + 1) + '.t7')
    
    
def compute_accuracy(config, model, data_loader, device):
    correct_pred, num_examples = 0, 0

    num_protected0, num_protected1 = 0, 0
    num_correct_pred_protected0, num_correct_pred_protected1 = 0, 0

    num_pred1_protected0, num_pred1_protected1 = 0, 0
    num_targets0_protected0, num_targets0_protected1 = 0, 0
    num_targets1_protected0, num_targets1_protected1 = 0, 0

    num_pred0_targets0_protected0, num_pred0_targets0_protected1 = 0, 0
    num_pred1_targets0_protected0, num_pred1_targets0_protected1 = 0, 0
    num_pred0_targets1_protected0, num_pred0_targets1_protected1 = 0, 0
    num_pred1_targets1_protected0, num_pred1_targets1_protected1 = 0, 0

    for i, (features, targets, protected) in enumerate(data_loader):
        if config['DEBUG'] and i > 1:
            break
        features = features.to(device, dtype=torch.float)
        targets = targets.to(device)
        protected = protected.to(device)
        _, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += 1.*targets.size(0)
        correct_pred += (predicted_labels == targets).float().sum()

        # For DDP metric
        num_protected0 += (protected == 0).float().sum()
        num_protected1 += (protected == 1).float().sum()
        num_correct_pred_protected0 += ((predicted_labels == targets)
                                        & (protected == 0)).float().sum()
        num_correct_pred_protected1 += ((predicted_labels == targets)
                                        & (protected == 1)).float().sum()

        # For PPV metric
        num_pred1_protected0 += ((predicted_labels == 1)
                                 & (protected == 0)).float().sum()
        num_pred1_protected1 += ((predicted_labels == 1)
                                 & (protected == 1)).float().sum()

        # For FPR metric
        num_targets0_protected0 += ((targets == 0)
                                    & (protected == 0)).float().sum()
        num_targets0_protected1 += ((targets == 0)
                                    & (protected == 1)).float().sum()

        # For FNR metric
        num_targets1_protected0 += ((targets == 1)
                                    & (protected == 0)).float().sum()
        num_targets1_protected1 += ((targets == 1)
                                    & (protected == 1)).float().sum()


        num_pred0_targets0_protected0 += ((predicted_labels == 0)
                                          & (targets == 0) & (protected == 0)).float().sum()
        num_pred0_targets0_protected1 += ((predicted_labels == 0)
                                          & (targets == 0) & (protected == 1)).float().sum()


        num_pred1_targets0_protected0 += ((predicted_labels == 1)
                                          & (targets == 0) & (protected == 0)).float().sum()
        num_pred1_targets0_protected1 += ((predicted_labels == 1)
                                          & (targets == 0) & (protected == 1)).float().sum()


        num_pred0_targets1_protected0 += ((predicted_labels == 0)
                                          & (targets == 1) & (protected == 0)).float().sum()
        num_pred0_targets1_protected1 += ((predicted_labels == 0)
                                          & (targets == 1) & (protected == 1)).float().sum()
                                                                                                                                      

        num_pred1_targets1_protected0 += ((predicted_labels == 1)
                                          & (targets == 1) & (protected == 0)).float().sum()
        num_pred1_targets1_protected1 += ((predicted_labels == 1)
                                          & (targets == 1) & (protected == 1)).float().sum()


    # res imported from utils
    res = get_res_dict()

    res['acc'] = correct_pred.float()/num_examples * 100

    res['ddp'] = abs(num_correct_pred_protected0 / (num_protected0 + 1e-6) -
                     num_correct_pred_protected1 / (num_protected1 + 1e-6)) * 100

    res['ppv'] = abs(num_pred1_targets1_protected0 / (num_pred1_protected0 + 1e-6) -
                     num_pred1_targets1_protected1 / (num_pred1_protected1 + 1e-6)) * 100

    res['fpr'] = abs(num_pred0_targets0_protected0 / (num_targets0_protected0 + 1e-6) -
                         num_pred0_targets0_protected1 / (num_targets0_protected1 + 1e-6)) * 100

    res['fnr'] = abs(num_pred1_targets1_protected0 / (num_targets1_protected0 + 1e-6) -
                         num_pred1_targets1_protected1 / (num_targets1_protected1 + 1e-6)) * 100

    res['tn_s0'] = num_pred0_targets0_protected0
    res['tn_s1'] = num_pred0_targets0_protected1
    res['fp_s0'] = num_pred1_targets0_protected0
    res['fp_s1'] = num_pred1_targets0_protected1
    res['fn_s0'] = num_pred0_targets1_protected0
    res['fn_s1'] = num_pred0_targets1_protected1
    res['tp_s0'] = num_pred1_targets1_protected0
    res['tp_s1'] = num_pred1_targets1_protected1

    return res


def print_stats(config, epoch, stats, stat_type=''):
    print("------------------------------------------------------------------")
    print("##################",stat_type, "##################")
    print('Epoch: %03d/%03d | Acc: %.3f%% |  Ddp: %.3f%% |  Ppv: %.3f%% |  Fpr: %.3f%% |  Fnr: %.3f%% ' % (
        epoch+1, config['NUM_EPOCHS'],
        stats['acc'],
        stats['ddp'],
        stats['ppv'],
        stats['fpr'],
        stats['fnr']))
    print('                 |  TN0: %d |  FP0: %d |  FN0: %d |  TP0: %d' % (
        stats['tn_s0'],
        stats['fp_s0'],
        stats['fn_s0'],
        stats['tp_s0']))
    print('                 |  TN1: %d |  FP1: %d |  FN1: %d |  TP1: %d' % (
        stats['tn_s1'],
        stats['fp_s1'],
        stats['fn_s1'],
        stats['tp_s1']))
    print("------------------------------------------------------------------")
