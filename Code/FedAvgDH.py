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
train_loader, train_loader_list, test_loader, _ = get_loaders(config['NUM_AGENTS'])

#Model
NUM_CLASSES = 2
GRAYSCALE = False
criterion = torch.nn.CrossEntropyLoss(reduce=False)


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

    for epoch in range(inner_runs):
        eta = eta * config['ETA_BETA']
        running_loss = 0.0
        total_size = 0

        model.train()
        for batch_idx, (features, targets, protected) in enumerate(train_loader):
            if config['DEBUG'] and batch_idx > 1:
                break

            features = features.to(device)
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

for epoch in range(config['NUM_EPOCHS']):
    layerwise_weights = []
    for agent in range(config['NUM_AGENTS']):
        model, _,_ = train(train_loader_list[agent], agent, epoch, config['AGENT_EPOCHS'], trained_weights) #10 epochs
        
        if agent == 0:
            #append weights of model
            layerwise_weights = model.state_dict()

        else:
            #take avg
            temp_weights = model.state_dict()
            for key in temp_weights:
                layerwise_weights[key] += temp_weights[key]

    for key in layerwise_weights:
        layerwise_weights[key] = layerwise_weights[key]/config['NUM_AGENTS']

    print('Central_Epoch: % 03d/%03d ' % (epoch + 1, config['NUM_EPOCHS']))
    
    trained_weights = layerwise_weights

    model_new, optimizer, loss1 = train(train_loader, -1, epoch, 1, trained_weights) #10 epochs
    val_loss.append(loss1.cpu().detach())

    # model.eval()
    with torch.set_grad_enabled(False):
        train_stats = compute_accuracy(config, model_new, train_loader, device)
        Acc_loss.append(train_stats['acc'].cpu().detach())
        EOpp_loss.append(train_stats['fnr'].cpu().detach())
        EO_loss.append(max(train_stats['fnr'], train_stats['fpr']).cpu().detach())
        AP_loss.append((train_stats['fnr'] + train_stats['fpr']).cpu().detach())
        print_stats(config, epoch, train_stats, stat_type='Train')

        test_stats = compute_accuracy(config, model_new, test_loader, device)
        print_stats(config, epoch, test_stats, stat_type='Test')

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    best_model = False
    if test_stats['acc'] >= config['MIN_ACC']:
        if test_stats['acc'] > config['MIN_ACC']:
            best_model = True
        else:
            if test_stats['fnr'] < config['MIN_FAIRLOSS']:
                best_model = True

    if (best_model or not (epoch+1) % 5) and config['SAVE_CKPT']:
        print("Saving Model")
        config['MIN_ACC'] = test_stats['acc']
        config['MIN_FAIRLOSS'] = test_stats['fnr']

        save_everything(epoch, model_new, optimizer, 1, test_stats, True, config,"FedAvgDH")