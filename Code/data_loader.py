import os
import random
import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from PIL import Image

# Hyper-params
BATCH_SIZE = 256
label_attr = 'gender'
protected_attr = 'age'

class FFHQDataset(Dataset):
    
    def __init__(self, df, transform=None, label_attr = 'gender', protected_attr = 'age'):
        
        self.img_names = df['file'].values
        self.y = df[label_attr].values
        self.p = df[protected_attr].values
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open('images/'+self.img_names[index])

        if self.transform is not None:
            img = self.transform(img)
            
        label = self.y[index]
        protected = self.p[index]
        return img, label, protected
        
    def __len__(self):
        return self.y.shape[0]


def prepare_dataset(label_attr = 'gender', protected_attr = 'age'):

    print('Using label attribute:', label_attr)

    df1 = pd.read_csv('ffhq.csv')
    df1.head()

    return df1
    
def get_loaders(splits):

    df = prepare_dataset(label_attr, protected_attr)
    
    custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                           transforms.Resize((128, 128)),
                                           transforms.ToTensor()])

    dataset = FFHQDataset(df, transform=custom_transform,
                                  label_attr=label_attr,
                                  protected_attr=protected_attr)
    
    train_dataset, test_dataset = random_split(dataset, [len(dataset)-int(len(dataset)*0.2), int(len(dataset)*0.2)])

    part_list = [int(len(train_dataset) / splits)] * (splits - 1)
    part_list.append(len(train_dataset) - int(len(train_dataset) / splits) * (splits - 1))
    
    train_splits = random_split(train_dataset, part_list)

    train_loader_full = DataLoader(dataset=train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4)
    train_loader_list = []
    for i in range(splits):
        train_loader = DataLoader(dataset=train_splits[i],
                                batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=4)
        train_loader_list.append(train_loader)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4)


    return train_loader_full, train_loader_list, test_loader