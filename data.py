## data.py Imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.models as models

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np


def dataloader_CIFAR10(args):
    batch_size = args['batch_size']
    num_workers = args['num_workers']
    root = args['data_root']
    valid_size = args['val_split_size']

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.15, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_val = transform_test

    ## Downloading and loading the dataset

    trainset = datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform_train)

    valset = datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform_val)

    testset = datasets.CIFAR10(root=root, train=False,
                                          download=True, transform=transform_test)

    ## Splitting for val

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    ## Load the dataset (## Sampler naturally gives shuffle)

    tr_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                              num_workers=num_workers)

    va_loader = DataLoader(valset, batch_size=batch_size, sampler=valid_sampler,
                                              num_workers=num_workers)

    te_loader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    
    return tr_loader, va_loader, te_loader



## Imagenet12 dataloader
def dataloader_IMAGENET12(args):
    batch_size = args['batch_size']
    image_size = args['model_input_size'] ## model input size and not image size
    num_workers = args['num_workers']
    root = args['data_root']
    valid_size = args['val_split_size']


    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    transform_val = transform_test

    trainset = datasets.ImageFolder(root + 'train/', transform=transform_train)
    valset = datasets.ImageFolder(root + 'train/', transform=transform_val)
    testset = datasets.ImageFolder(root + 'val/', transform=transform_test)

    ## Splitting for val

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    ## Load the dataset (## Sampler naturally gives shuffle)

    tr_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                           num_workers=num_workers)

    va_loader = DataLoader(valset, batch_size=batch_size, sampler=valid_sampler,
                           num_workers=num_workers)

    te_loader = DataLoader(testset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    return tr_loader, va_loader, te_loader
