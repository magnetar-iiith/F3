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

    def __init__(self, df, transform=None, label_attr='gender', protected_attr='age'):

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


def prepare_dataset(label_attr = 'gender', protected_attr = 'race'):

    print('Using label attribute:', label_attr)

    df = pd.read_csv('ffhq.csv')

    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.9
    df2 = df[msk]
    test = df[~msk]

    df2['split'] = np.random.randn(df2.shape[0], 1)
    msk = np.random.rand(len(df2)) <= 0.8
    train = df2[msk]
    val = df2[~msk]


    df0 = train.loc[train[protected_attr] == 0]
    df1 = train.loc[train[protected_attr] == 1]

    dataset = []
    dataset.append(df0)  # 0.42
    dataset.append(df1)  # 0.19

    return train, test, val, dataset
    
def get_loaders(splits):

    train, test, val, df_list = prepare_dataset(label_attr, protected_attr)
    
    custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                           transforms.Resize((128, 128)),
                                           transforms.ToTensor()])

    test_dataset = FFHQDataset(test,
                        transform=custom_transform,
                        label_attr=label_attr,
                        protected_attr=protected_attr)

    train_dataset = FFHQDataset(train,
                    transform=custom_transform,
                    label_attr=label_attr,
                    protected_attr=protected_attr)

    val_dataset = FFHQDataset(val,
                        transform=custom_transform,
                        label_attr=label_attr,
                        protected_attr=protected_attr)

    dataset_full = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    
    dataset = []
    for i in range(2):
        dataset.append(FFHQDataset(df_list[i],transform=custom_transform, label_attr=label_attr, protected_attr=protected_attr))
    
    print("size of diff age groups", len(dataset[0]), len(dataset[1]))
    s = [0]*2
    s[0] = int(splits * 0.5)
    s[1] = int(splits - s[0])
    print("samples per network", int(len(dataset[0]) / s[0]), int(len(dataset[1]) / s[1]))
    
    train_splits = []
    for i in range(2):
        part_list = [int(len(dataset[i]) / s[i])] * (s[i] - 1)
        part_list.append(len(dataset[i]) - int(len(dataset[i]) / s[i]) * (s[i] - 1))
        train_splits.append(random_split(dataset[i], part_list))

    train_loader_full = DataLoader(dataset=dataset_full,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=4)
    train_loader_list = []
    for i in range(2):
        for j in range(s[i]):
            train_loader = DataLoader(dataset=train_splits[i][j],
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=4)
            train_loader_list.append(train_loader)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=4)


    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4)


    return train_loader_full, train_loader_list, test_loader, valid_loader