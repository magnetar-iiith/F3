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
lam = 0.7
NUM_CLASSES = 2
GRAYSCALE = False
criterion = torch.nn.CrossEntropyLoss(reduce=False)


def train(train_loader, agent, out_epoch, wts=None):
    
    model1 = resnet18(NUM_CLASSES, GRAYSCALE)
    model2 = resnet18(NUM_CLASSES, GRAYSCALE)

    if wts != None:
        print("updating params")
        model1.load_state_dict(wts)
        model2.load_state_dict(wts)

    model1 = model1.to(device)
    model2 = model2.to(device)

    if agent == -1:
        return model1, 0

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=config['LR'])
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=config['LR'])

    model1.train()
    for epoch in range(config['AGENT_EPOCHS']):
        acc_loss = 0.0
        fair_loss = 0.0
        val_size = 0.0
        total_size = 0
        epoch_fair_loss = 0
        
        if epoch != 0:
            for _, (features, targets, protected) in enumerate(valid_loader):

                features = features.to(device)
                targets = targets.to(device)
                protected = protected.to(device)

                # The Inner loop helps in finding a good local minima
                output, _ = model1(features)
                loss_all = criterion(output, targets)
                loss_t1_s0 = loss_all[(targets == 1) & (protected == 0)].mean()
                loss_t1_s1 = loss_all[(targets == 1) & (protected == 1)].mean()
                loss_t0_s0 = loss_all[(targets == 0) & (protected == 0)].mean()
                loss_t0_s1 = loss_all[(targets == 0) & (protected == 1)].mean()

                loss2 = max(abs(loss_t1_s0 - loss_t1_s1), abs(loss_t0_s0 - loss_t0_s1))
                fair_loss += loss2 * features.size(0)
                val_size += features.size(0)

                optimizer1.zero_grad()
                loss2.backward()
                optimizer1.step()
            epoch_fair_loss = fair_loss/val_size

        for _, (features, targets, protected) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)
            protected = protected.to(device)

            # The Inner loop helps in finding a good local minima
            output, _ = model2(features)
            loss_all = criterion(output, targets)

            loss1 = loss_all.mean()

            optimizer2.zero_grad()

            acc_loss += loss1 * features.size(0)
            total_size += features.size(0)

            # loss = (lam) * loss1.clone()

            loss1.backward()

            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                if epoch == 0:
                    avg = p2.grad
                else:
                    avg = (1- lam) * p1.grad + (lam) * p2.grad
                p1.grad = avg
                p2.grad = avg.clone()

            optimizer2.step()

        epoch_acc_loss = acc_loss/total_size
        model1 = model2

        ### LOGGING
        print('Central_Epoch: % 03d/%03d | Agent: % 02d/%02d | Epoch: % 03d/%03d | Acc_loss: % .4f | Fair_loss: % .4f' % (out_epoch+1, config['NUM_EPOCHS'], agent+1, config['NUM_AGENTS'], epoch+1, config['AGENT_EPOCHS'], epoch_acc_loss, epoch_fair_loss))

    return model1, epoch_acc_loss

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
        model, _ = train(train_loader_list[agent], agent, epoch, trained_weights) #10 epochs
        
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
    model_new, _ = train(train_loader, -1, epoch, trained_weights)  #10 epochs


    model_new.eval()
    with torch.set_grad_enabled(False):
        train_stats = compute_accuracy(config, model_new, train_loader, device)
        Acc_loss.append(train_stats['acc'].cpu().detach())
        EOpp_loss.append(train_stats['fnr'].cpu().detach())
        EO_loss.append(max(train_stats['fnr'], train_stats['fpr']).cpu().detach())
        AP_loss.append((train_stats['fnr'] + train_stats['fpr']).cpu().detach())
        print_stats(config, epoch, train_stats, stat_type='Train')

        if epoch < 20:
            best_model = True

        elif (train_stats['acc'] - best_acc >= 1):
            best_model = True

        elif (train_stats['acc'] - best_acc >= 0) and max(train_stats['fnr'], train_stats['fpr']) < best_EO:
            best_model = True

        test_stats = compute_accuracy(config, model_new, test_loader, device)
        print_stats(config, epoch, test_stats, stat_type='Test')

    if best_model and config['SAVE_CKPT']:
        print("Saving Model on Epoch ", epoch+1)

        best_acc = train_stats['acc']
        best_EO = max(train_stats['fnr'], train_stats['fpr'])
        best_model = False

    if epoch > 20 and max(Acc_loss[-19:]) - Acc_loss[-20] < 1:
        break


print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
print('Stopped at epoch - ', epoch + 1)
