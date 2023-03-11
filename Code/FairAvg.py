import os
import time

import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image
import yaml

#Helper Functions
from DH_data_loader import get_loaders
from model import resnet18
from helper import *

#SET PARAMS
with open('param.yml') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load Data
train_loader, train_loader_list, test_loader, valid_loader = get_loaders(config['NUM_AGENTS'])

#Model
K = 1
NUM_CLASSES = 2
GRAYSCALE = False
criterion = torch.nn.CrossEntropyLoss(reduce=False)


def Nmaxelements(performance_list, models, N):
    model_list = []

    for _ in range(0, N):
        m1 = []
        acc1 = 0
        fnr1 = 100
        idx = 0

        for j in range(len(performance_list)):
            if max(performance_list[j]['fnr'], performance_list[j]['fpr']) < fnr1 or (max(performance_list[j]['fnr'], performance_list[j]['fpr']) == fnr1 and performance_list[j]['acc'] > acc1):
                train_stats = performance_list[j]
                m1 = models[j]
                fnr1 = max(performance_list[j]['fnr'], performance_list[j]['fpr'])
                acc1 = performance_list[j]['acc']
                idx = j

        performance_list.remove(train_stats)
        models.pop(idx)
        model_list.append(m1)

    return model_list

def train(train_loader, agent, out_epoch, inner_runs, wts=None):
    
    model = resnet18(NUM_CLASSES, GRAYSCALE)

    if wts != None:
        print("updating params")
        model.load_state_dict(wts)

    model = model.to(device)
    
    lam0 = config['LAM0_PRIOR']
    lam1 = config['LAM1_PRIOR']
    eta = config['ETA_INIT']

    optimizer = torch.optim.SGD(model.parameters(), lr=config['LR']) #adam

    if agent == -1:
        return model, optimizer, 0

    for epoch in range(inner_runs):
        running_loss = 0.0
        total_size = 0

        model.train()
        for batch_idx, (features, targets, protected) in enumerate(train_loader):
            if config['DEBUG'] and batch_idx > 1:
                break

            features = features.to(device, dtype=torch.float)
            targets = targets.to(device)
            protected = protected.to(device)

            # The Inner loop helps in finding a good local minima
            for _ in range(config['NUM_INNER']):
                output, _ = model(features)
                loss_all = criterion(output, targets)
                loss_t1_s0 = loss_all[(targets == 1) & (protected == 0)].mean()
                loss_t1_s1 = loss_all[(targets == 1) & (protected == 1)].mean()

                train_loss = loss_all.mean()
                loss2 = abs(loss_t1_s0 - loss_t1_s1)

                # Primal Update
                optimizer.zero_grad()
                loss = loss_all.mean()
                loss.backward()
                optimizer.step()

                running_loss += loss * features.size(0)
                total_size += features.size(0)

            ### LOGGING
            if not batch_idx % 50:
                print('Central_Epoch: % 03d/%03d | Agent: % 02d/%02d | Epoch: % 03d/%03d | Batch % 04d/%04d | train_loss: % .4f | loss2: % .4f' % (out_epoch+1, config['NUM_EPOCHS'], agent+1, config['NUM_AGENTS'], epoch+1, inner_runs, batch_idx+1,
                                                                                                    len(train_loader), train_loss, loss2))
                print('eta: %.3f | lam0: %.3f | lam1: %.3f' % (eta, lam0, lam1))

    epoch_loss = running_loss / total_size
    return model, optimizer, epoch_loss

print(config)

start_time = time.time()

#TRAIN
trained_weights = None
val_loss = []
Acc_loss = []
EOpp_loss = []
EO_loss = []
AP_loss = []
best_acc = 0
best_model = False

global_model = []
for epoch in range(config['NUM_EPOCHS']):
    performance_list = []
    model_list = []


    for agent in range(config['NUM_AGENTS']):
        model, _, _ = train(train_loader_list[agent], agent, epoch, config['AGENT_EPOCHS'], trained_weights) #10 epochs
        
        train_stats = compute_accuracy(config, model, valid_loader, device)

        if agent == 0:
            global_model = model.state_dict()
            model_weights = global_model

        else:
            model_weights = model.state_dict()

        performance_list.append(train_stats)
        
        m = []
        for key in model_weights:
            m.append(model_weights[key])
        model_list.append(m)

    final_models = Nmaxelements(performance_list, model_list, K)

    m1 = global_model
    idx = 0
    for key in m1:
        m1[key] = final_models[0][idx]
        idx = idx+1

    for i in range(1, K):
        idx = 0
        for key in m1:
            m1[key] += final_models[i][idx]
            idx = idx + 1

    for key in m1:
        m1[key] = m1[key]/K

    print('Central_Epoch: % 03d/%03d ' % (epoch + 1, config['NUM_EPOCHS']))
    
    trained_weights = m1

    model_new, optimizer, _ = train(train_loader, -1, epoch, 1, trained_weights) #10 epochs

    with torch.set_grad_enabled(False):
        train_stats = compute_accuracy(config, model_new, train_loader, device)
        print_stats(config, epoch, train_stats, stat_type='Train')
        val_loss.append((train_stats['acc']).cpu().detach())

        if epoch < 20:
            best_model = True

        elif (train_stats['acc'] - best_acc >= 1):
            best_model = True

        elif (train_stats['acc'] - best_acc >= 0) and max(train_stats['fnr'], train_stats['fpr']) < best_EO:
            best_model = True

        test_stats = compute_accuracy(config, model_new, test_loader, device)
        print_stats(config, epoch, test_stats, stat_type='Test')
        Acc_loss.append(train_stats['acc'].cpu().detach())
        EOpp_loss.append(test_stats['fnr'].cpu().detach())
        EO_loss.append(max(test_stats['fnr'].cpu().detach(), test_stats['fpr'].cpu().detach()))
        AP_loss.append(test_stats['fnr'].cpu().detach() + test_stats['fpr'].cpu().detach())

    if best_model and config['SAVE_CKPT']:
        print("Saving Model on Epoch ", epoch+1)

        best_acc = train_stats['acc']
        best_EO = max(train_stats['fnr'], train_stats['fpr'])
        best_model = False

        save_everything(1, model_new, optimizer, 1, test_stats, True, config, '{0}-FairAvg'.format(K))

    # if epoch > 20 and max(val_loss[-19:]) - val_loss[-20] < 1:
    #     break

print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

print('Stopped at epoch - ', epoch+1)