import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# from dataloader.transform import PretrainTransform
from dataloader.transform import PretrainTransform_FSCIL

class ExemplarSet(Dataset):
    def __init__(self, exemplar_path, exemplar_label,args ,dataset='mini_imagenet', train=False):
        self.dataset = dataset
        self.train = train  # training or testing stage
        self.return_idx = False
        if self.dataset == 'mini_imagenet':
            image_size = 84
            self.transform_train = PretrainTransform_FSCIL('mini_imagenet',args)
            self.transform_test = transforms.Compose([
                transforms.Resize([96, 96]),
                transforms.CenterCrop(image_size),
                # transforms.RandomResizedCrop(84, scale=(0.9,1)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        elif self.dataset == 'cub200':
            self.transform_train = PretrainTransform_FSCIL('cub200', args)
            self.transform_test = transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224, scale=(0.25,1), ratio=(1,1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = PretrainTransform_FSCIL('cifar100', args)
            self.transform_test = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
            ])
            
        
        self.update(exemplar_path, exemplar_label)
        
    def update(self, exemplar_path, exemplar_label):
        self.data = exemplar_path
        self.targets = exemplar_label
        
    def add(self, exemplar_path, exemplar_label):
        self.data.append(exemplar_path)
        self.targets.append(exemplar_label)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, targets = self.data[i], self.targets[i]
        if self.train:
            if self.dataset == 'cifar100':
                image = self.transform_train(Image.fromarray(img))
            else:
                image = self.transform_train(Image.open(img).convert('RGB'))
        else:
            if self.dataset == 'cifar100':
                image = self.transform_train(Image.fromarray(img))
            else:
                image = self.transform_test(Image.open(img).convert('RGB'))
        
        return image,targets


class Batch_DataSampler():
    def __init__(self, label, n_batch, n_query, ep_per_batch=1):
        self.n_batch = n_batch  # number of episode
        self.n_query = n_query  # number of query samples to sample from base classes
        self.ep_per_batch = ep_per_batch
        self.label = np.array(label) 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                l = np.random.choice(len(self.label), self.n_query, replace=True)
                batch.append(torch.from_numpy(l))
            batch = torch.stack(batch) # bs * n_query 
            yield batch.view(-1)

class Batch_DataSampler_catAll():
    def __init__(self, label, n_batch, n_query=0, ep_per_batch=1):
        self.n_batch = n_batch  # number of episode
        self.n_query = n_query  # number of query samples to sample from base classes
        self.ep_per_batch = ep_per_batch
        self.label = np.array(label) 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                dataset_len = len(self.label)
                seq = torch.arange(dataset_len)
                
                #打乱seq
                shuffle_indices = torch.randperm(dataset_len)
                shuffle_seq = seq[shuffle_indices]
                
                # l = np.random.choice(len(self.label), self.n_query, replace=True)
                # batch.append(torch.from_numpy(l))
                batch.append(shuffle_seq)
            batch = torch.stack(batch) # bs * n_query 
            yield batch.view(-1)

