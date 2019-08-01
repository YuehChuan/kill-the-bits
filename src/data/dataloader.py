# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_data(data_path='', batch_size=128, nb_workers=64):
    """
    Loads data from ImageNet dataset.

    Args:
        - data_path: path to dataset
        - batch_size: train and test batch size
        - nb_workers: number of workers for dataloader
    """
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    trainset = CIFAR10(root=data_path,
                        train=True,
                        download=False,
                        transform=transform_train)

    testset = CIFAR10(root=data_path,
                       train=False,
                       download=False,
                       transform=transform_test)


    train_loader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    test_loader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    """


    trainset = CIFAR100(root=data_path,
                        train=True,
                        download=False,
                        transform=transform_train)

    testset = CIFAR100(root=data_path,
                       train=False,
                       download=False,
                       transform=transform_test)

    # data path
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transf_train = transforms.Compose([
                 transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize])
    transf_test = transforms.Compose([
                 transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 normalize
                 ])

    train_set = datasets.ImageFolder(
            root=train_data_path, transform=transf_train)
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=nb_workers, pin_memory=True, drop_last=True)

    test_set = datasets.ImageFolder(
            root=test_data_path, transform=transf_test)
    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=nb_workers, pin_memory=True)
    """
