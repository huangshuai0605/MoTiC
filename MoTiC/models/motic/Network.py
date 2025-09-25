import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchvision.models.resnet import resnet18
from models.resnet20_cifar import resnet20


from torchvision import transforms

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # #scale_cos
        # self.scale_cos = nn.Parameter( torch.FloatTensor(1).fill_(16.0),requires_grad=True )
        
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.feature_dim = 64
            args.feature_dim = 64
            # self.classifier = nn.Linear(self.feature_dim, self.args.num_classes, bias=False)

            self.num_features = 640
            
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)
            self.encoder.fc = nn.Identity()
            self.feature_dim = 512
            args.feature_dim = 512
            # self.classifier = nn.Linear(self.feature_dim, self.args.num_classes, bias=False)

        elif self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.feature_dim = 512
            args.feature_dim = 512
            self.num_features = 512
            self.encoder.fc = nn.Identity()

        if args.fantasy == "rotation":
            m = 4
        elif args.fantasy == "rotation2":
            m = 2
        else:
            m = 12 
        
        # self.classifier = nn.Linear(self.feature_dim, self.args.num_classes, bias=False)
        self.classifier = nn.Linear(self.feature_dim, self.args.num_classes*m, bias=False)
        
        self.session = 0
        self.test = False
    
    def forward_metric(self, x, pos=None):
        inp = x

        x = self.encoder(x)
        fc = self.classifier.weight

        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
        x = self.args.temperature * x
        # x = self.scale_cos * x
        
        return x

    def encode(self, x):
        x = self.encoder(x) # (b, c, h, w)
        # x = F.adaptive_avg_pool2d(x, 1)
        # x = x.squeeze(-1).squeeze(-1)
        return x

    
    
    def forward(self, input, pos=None):
        if self.mode != 'encoder':
            input = self.forward_metric(input,pos)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    # def update_fc(self,dataloader,class_list,transform,session):
    #     for batch in dataloader:
    #         data, label = [_.cuda() for _ in batch]
    #         b = data.size()[0]
    #         data = transform(data)
    #         m = data.size()[0] // b 
    #         labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
    #         # data, _ =self.encode_q(data)
    #         data = self.encoder(data)
    #         data.detach()

    #     if self.args.not_data_init:
    #         new_fc = nn.Parameter(
    #             torch.rand(len(class_list)*m, self.num_features, device="cuda"),
    #             requires_grad=True)
    #         import math
    #         nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
    #     else:
    #         new_fc = self.update_fc_avg(data, labels, class_list, m)
    
    def update_fc(self,dataloader,class_list,transform,session):
        feats = []
        labels = []
        with torch.no_grad():
            if self.args.dataset == 'cifar100':
                size = 32
                dataloader.dataset.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))])
            elif self.args.dataset == 'mini_imagenet':
                size = 84
                dataloader.dataset.transform = transforms.Compose([
                    transforms.Resize([96, 96]),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
            else:
                size = 224
                dataloader.dataset.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            for batch in dataloader:
                data, label = [_.cuda() for _ in batch]
                b = data.size()[0]
                data = transform(data)
                m = data.size()[0] // b
                label_transform = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
                
                feats.append(self.encoder(data).detach())

                labels.append(label_transform)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels,dim=0)

        for ii in range(labels.unique().shape[0]):
            self.classifier.weight.data[labels.min()+ii] = feats[labels==labels.min()+ii].mean(dim=0)

    def SelfSupervised(self,feat1,feat2):
        sim_matrix = torch.mm(feat1,feat2.T)
        sim_matrix /= 0.07
        labels = torch.arange(0,feat1.shape[0], step=1, dtype=torch.long).cuda(self.args.gpu)
        loss_contrast1 = F.cross_entropy(sim_matrix, labels)
        loss_contrast2 = F.cross_entropy(sim_matrix.T,labels)
        
        loss_contrast = (loss_contrast1 + loss_contrast2)/2
        return loss_contrast
    
class MocoNet(nn.Module):
    def __init__(self, args, encoder_q,classifier):
        super(MocoNet, self).__init__()
        self.args = args
        self.m = 0.999
        if args.dataset == 'mini_imagenet':
            num = int(8192/args.batch_size_base)   #8192
        elif args.dataset == 'cub200': 
            num = int(8192/args.batch_size_base)    #8192
        elif args.dataset == 'cifar100':
            num = int(65536/args.batch_size_base)   #60000
        self.K = args.batch_size_base*num
        self.T = args.ssc_temp
        
        args.K = self.K
        
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        #classifier
        self.classifier = classifier
        
        # #scale cosine
        # self.scale_cos = scale_cos
        
        #resNet 18
        self.encoder_q = encoder_q  
        if self.args.dataset in ['mini_imagenet']:
            self.encoder_k = resnet18(False, args)
            self.encoder_k.fc = nn.Identity()
        elif self.args.dataset == 'cub200':
            self.encoder_k = resnet18(True, args)
            self.encoder_k.fc = nn.Identity()
        elif self.args.dataset in ['cifar100']:
            self.encoder_k = resnet20()
        self.encoder_k.load_state_dict(encoder_q.state_dict(), strict=True)
        
        #train classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        #gradient encoder_q, batchNorm  work
        for param in self.encoder_q.parameters():
            param.requires_grad = True
        self.encoder_q.train()
        
        #freeze encoder_k, batchNorm work
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.encoder_k.train()
        
        #create the queue
        #创造队列字典
        self.register_buffer("queue", torch.randn(args.feature_dim,self.K) )     #[512, queue_size]
        self.queue = nn.functional.normalize(self.queue,dim=0)  #按列进行归一化
        #创建一个队列指针
        self.register_buffer("queue_ptr", torch.zeros(1,dtype=torch.long) )
        #创建一个队列label
        self.register_buffer("queue_label", torch.zeros(self.K,dtype=torch.uint8) )
    
    def train_mode(self):
        self.encoder_q.train()  #batchNorm work
    
    def eval_mode(self):
        self.encoder_q.eval()   #batchNorm eval
    
    @torch.no_grad()    
    def _momentum_update_key_encoder(self):
        """
        Momentum update the key encoder
        """    
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    #队列的特征添加与删除，用替换和指针实现
    def _dequeue_and_enqueue(self, keys,train_label):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        assert self.K % batch_size == 0
        
        self.queue[:,ptr:ptr + batch_size] = keys.T
        self.queue_label[ptr:ptr+batch_size] = train_label
        #将队列指针移动
        ptr = (ptr + batch_size) % self.K
        #更新指针位置
        self.queue_ptr[0] = ptr
    
    #计算类间相似性的损失
    def cal_loss_class_external(self,q, train_label):
        #mask: [B,K] 
        mask = (train_label.unsqueeze(1) != self.queue_label.unsqueeze(0)).float()
        #[B,C] * [C,K] -> [B,K]
        sim_matrix = torch.mm(q, self.queue )
        
        
        num_noteq_class = (mask==1).sum()
        loss_class_exter = (sim_matrix *mask ).sum()/num_noteq_class
        return loss_class_exter
    
        
        
    #前向过程
    def forward(self, im_q1, im_q2 ,im_k, train_label, cal_exter_loss):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        #compute query features
        q1 = self.encoder_q(im_q1)
        q1 = nn.functional.normalize(q1,dim=1)    #按行归一化
        
        #compute query features
        q2 = self.encoder_q(im_q2)
        q2 = nn.functional.normalize(q2,dim=1)    #按行归一化
        
        #compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            if self.args.dataset == 'cub200':
                k = self.encoder_k(im_k)
                k = nn.functional.normalize(k,dim=1)
            else:
                #shuffle BN
                batch_size_all = im_k.shape[0]
                idx_shuffle = torch.randperm(batch_size_all).cuda()
                im_k_shuffle = im_k[idx_shuffle]
                k = self.encoder_k(im_k_shuffle)
                k = nn.functional.normalize(k, dim=1)   #归一化
                #还原顺序
                idx_unshuffle = torch.argsort(idx_shuffle)
                k = k[idx_unshuffle]  #还原k的顺序
            
        
        #postive logits:Nx1
        l_pos = torch.bmm(q1.view(q1.shape[0],1,-1), k.view(k.shape[0],-1,1) ).squeeze(-1)
        #negative logits:NxK
        l_neg = torch.mm(q1, self.queue.clone().detach() ) 
        #logits: Nx(1+K)
        logits = torch.cat([l_pos,l_neg],dim=1)
        #apply temperature
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss_contrast1 = self.criterion_ce(logits, labels)
        
        #postive logits:Nx1
        l_pos = torch.bmm(q2.view(q2.shape[0],1,-1), k.view(k.shape[0],-1,1) ).squeeze(-1)
        #negative logits:NxK
        l_neg = torch.mm(q2, self.queue.clone().detach() ) 
        #logits: Nx(1+K)
        logits = torch.cat([l_pos,l_neg],dim=1)
        #apply temperature
        logits /= self.T
        
        loss_contrast2 = self.criterion_ce(logits, labels)
        
        
        #dequeue and enqueue
        self._dequeue_and_enqueue(k,train_label)
        
        if cal_exter_loss:        
            loss_class_external1 = self.cal_loss_class_external(q1,train_label)
            loss_class_external2 = self.cal_loss_class_external(q2,train_label)
            return (loss_contrast1+loss_contrast2)/2, (loss_class_external1 + loss_class_external2)/2
        else:
            return (loss_contrast1+loss_contrast2)/2
        
        