# import new Network name here and add in model_class args
from .Network import *
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pdb
import math


def warmup_cosine(optimizer, current_epoch, max_epoch,lr_min = 0.0, lr_max = 0.1, warmup_epoch=10):
    eps = 1e-9
    for param_group in optimizer.param_groups:
        #假设每个参数组可以有自己的lr_min, lr_max 和warmup_epoch
        lr_min_group = param_group.get('lr_min', lr_min)
        lr_max_group = param_group.get('lr_max', lr_max)
        warmup_epoch_group = param_group.get('warmup_epoch', warmup_epoch)
        
        if current_epoch < warmup_epoch_group:
            lr = (lr_max_group * (current_epoch + 1) /  warmup_epoch_group) + eps
        else:
            lr = lr_min_group + 0.5*(lr_max_group - lr_min_group)*(1 + math.cos(math.pi*(current_epoch-warmup_epoch_group)/(max_epoch -warmup_epoch_group) ) ) + eps
        #更新当前参数组的学习率
        param_group["lr"] = lr

def base_train(MocoNet, trainloader, optimizer,  epoch, args, transform, num_trans):
    tl = Averager()
    ta = Averager()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)

    warmup_cosine(optimizer, epoch, args.epochs_base, lr_min=0,lr_max=0 ,warmup_epoch=0 )

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_ for _ in batch]

        B,C,H,W = data[0].shape     #batch_size, channel, height, weight
        num_aug = args.num_aug
        data[0] = data[0].cuda()    #[B,C,H,W]   #train_label
        data[1] = data[1].cuda()
        data[2] = data[2].cuda()
        
        data[0] = transform(data[0])  #[2*B,C,H,W]  #语义变换trainlabel
        data[1] = transform(data[1])
        data[2] = transform(data[2])
        train_label = train_label.cuda()
        
        m = num_trans
        joint_labels = torch.stack([train_label*m+ii for ii in range(m)],1 ).view(-1)   #语义变换trainlabel
        
        if epoch >= 1:
            loss_contrast, loss_class_external = MocoNet(data[0],data[1],data[2],joint_labels,True)
        elif epoch == 0:
            loss_contrast = MocoNet(data[0],data[1],data[2],joint_labels,False)
        
        data = torch.cat(data,dim=0).cuda()     #将num_aug个data合在一起
        feats = F.normalize(MocoNet.encoder_q(data),dim=-1 )
        # train_label = train_label.cuda()
        
        logits = F.linear( feats, F.normalize(MocoNet.classifier.weight[:args.base_class*m],dim=-1)   )  * args.temp
        
        ce_loss = F.cross_entropy(logits,joint_labels.repeat(num_aug+1))
        
        # ce_loss = F.cross_entropy(logits, train_label.repeat(num_aug+1)) 
        split_logits = torch.split(logits, split_size_or_sections=m*B, dim=0)
        #聚合预测 得学一学     #joint_preds:[2B,args.base_class*m]
        agg_preds_all = []
        for i in range(num_aug+1):
            split_l = split_logits[i]
            agg_preds = 0
            for j in range(m):    #聚合所有变换类预测：[B,args.base_class]
                agg_preds = agg_preds + split_l[j::m, j::m] / m
            agg_preds_all.append(agg_preds)
        
        agg_preds_all = torch.cat(agg_preds_all,dim=0)

        if epoch >= 1:
            loss = ce_loss + args.ssc_lamb*loss_contrast - args.inter_lamb*loss_class_external 
        elif epoch == 0:
            loss = ce_loss + args.ssc_lamb*loss_contrast
        acc = count_acc(agg_preds_all, train_label.repeat(num_aug+1))
        # acc = count_acc(logits, train_label.repeat(logits.shape[0]//B))
        

        # lrc = scheduler.get_last_lr()[0]
        lrc = optimizer.param_groups[0]['lr']
        if epoch>=1:
            tqdm_gen.set_description(
                'Session 0,epo{},lrc={:.4f},loss_ce={:.3f},loss_contrast={:.3f},loss_cl_ext={:.3f},total_loss={:.3f},acc={:.4f}'.format(epoch, lrc, ce_loss.item(),args.ssc_lamb*loss_contrast.item(),args.inter_lamb*loss_class_external.item(),loss.item(),acc)
            )
        else:
            tqdm_gen.set_description(
                'Session 0,epo{},lrc={:.4f},loss_ce={:.3f},loss_contrast={:.3f},total_loss={:.3f},acc={:.4f}'.format(epoch, lrc, ce_loss.item(),loss_contrast.item(),loss.item(),acc)
            )
        
        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()
    return tl, ta

def cal_loss_class(args,normalized_baseEmbedding, train_label):
    mask = torch.eq(train_label.unsqueeze(1), train_label.unsqueeze(0)).float().cuda(args.gpu)  #labels相等为1.0，不相等为0.0
    mask = mask*2 -1 #labels相等为1.0，不相等为-1
    mask = torch.ones(mask.shape).triu(diagonal=1).cuda(args.gpu) *mask  #保留主对角线上的
    
    cosine_matrix = normalized_baseEmbedding @ normalized_baseEmbedding.T
    num_eq_class = (mask==1).sum()
    num_noteq_class = (mask==-1).sum()
    
    #防止除以0.0
    if np.isclose(num_eq_class.item(),0.0 ):        #没有相等的label
        loss_class_inter = 0.0
    else:
        loss_class_inter = ( cosine_matrix*(mask==1) ).sum() / num_eq_class   #类内向量相似损失
    
    if np.isclose(num_noteq_class.item(),0.0 ):     #label全相等
        loss_class_external = 0.0       #
    else:
        loss_class_external = ( cosine_matrix*(mask==-1) ).sum() / num_noteq_class #类外向量相似损失
        
    return loss_class_inter, loss_class_external

# def replace_base_fc(trainset, transform, model, args):
#     # replace fc.weight with the embedding average of train data
#     model = model.eval()

#     trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
#                                               num_workers=8, pin_memory=True, shuffle=False)
#     trainloader.dataset.transform = transform  #注意一下，更换成test_dataloader的transform了
#     embedding_list = []
#     label_list = []
#     # data_list=[]
#     with torch.no_grad():
#         for i, batch in enumerate(trainloader):
#             data, label = [_.cuda() for _ in batch]
#             model.mode = 'encoder'
#             #embedding = model.module.encoder(data)
#             embedding = model.encoder(data)
#             embedding_list.append(embedding.cpu())
#             label_list.append(label.cpu())

#     embedding_list = torch.cat(embedding_list, dim=0)
#     label_list = torch.cat(label_list, dim=0)

#     proto_list = []

#     for class_index in range(args.base_class):
#         data_index = (label_list == class_index).nonzero()
#         embedding_this = embedding_list[data_index.squeeze(-1)]
#         embedding_this = embedding_this.mean(0)
#         proto_list.append(embedding_this)

#     proto_list = torch.stack(proto_list, dim=0)
    
#     model.classifier.weight.data[:args.base_class] = proto_list

#     return model

def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class*m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    # model.fc.weight.data[:args.base_class*m] = proto_list
    model.classifier.weight.data[:args.base_class*m] = proto_list
    return model

@torch.no_grad()
def test(model, testloader, epoch, transform, args, session):
    test_class = args.base_class + session * args.way  
    model = model.eval()
    vl = Averager()
    va_total = 0
    va_correct = 0
    if session > 0:
        va_base_total = 0
        va_base_correct = 0
        va_new_total = 0
        va_new_correct = 0

    model.session = session
    model.test = True

    # labels = []
    pred = []        
    # accs = np.zeros([100])
    # feats = []

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b                        
            # logits = model(data)
            # logits = logits[:, :test_class]
            joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class*m]
            
            agg_preds = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            loss = F.cross_entropy(agg_preds, test_label)
            # acc = count_acc(agg_preds,test_label)
            # loss = F.cross_entropy(logits, test_label)
            # acc = count_acc(logits, test_label)
            # pred = logits.argmax(dim=-1)
            pred = agg_preds.argmax(dim=-1)
            
            correct = pred == test_label

            vl.add(loss.item())
            va_total += pred.shape[0]
            va_correct += correct.sum()

            if session > 0:
                # pred = logits.argmax(dim=-1)
                pred = agg_preds.argmax(dim=-1)
                correct = pred == test_label

                base_mask = test_label < args.base_class
                new_mask = test_label >= args.base_class

                va_base_total += base_mask.sum()
                va_new_total += new_mask.sum()

                va_base_correct += correct[base_mask].sum()
                va_new_correct += correct[new_mask].sum()
            

        vl = vl.item()
        va = va_correct / va_total
        if session >0:
            va_base = va_base_correct / va_base_total
            va_new = va_new_correct / va_new_total

            assert va_total == va_base_total + va_new_total
            assert va_correct == va_base_correct + va_new_correct
        
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    if session > 0:
        return vl, va, va_base, va_new
    else:
        return vl, va

# def test(model, testloader, epoch, transform, args, session):
#     test_class = args.base_class + session * args.way
#     model = model.eval()
#     vl = Averager()
#     va = Averager()
#     with torch.no_grad():
#         tqdm_gen = tqdm(testloader)
#         for i, batch in enumerate(tqdm_gen, 1):
#             data, test_label = [_.cuda() for _ in batch]
#             b = data.size()[0]
#             data = transform(data)
#             m = data.size()[0] // b
#             joint_preds = model(data)
#             joint_preds = joint_preds[:, :test_class*m]
            
#             agg_preds = 0
#             for j in range(m):
#                 agg_preds = agg_preds + joint_preds[j::m, j::m] / m
            
#             loss = F.cross_entropy(agg_preds, test_label)
#             acc = count_acc(agg_preds, test_label)

#             vl.add(loss.item())
#             va.add(acc)

#         vl = vl.item()
#         va = va.item()
#     print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

#     return vl,va

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif (labels is None and mask is None): #true
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)   #[128,128]
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]      #3个增强视图
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)      #每个样本三个增强版本， features:[128,3,512] ->3 个 [128,512] ->concat起来 [384,512]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':           #true
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(        # [384,512] * [512,384] -> [384,384]/t 
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) #每一行取最大值
        
        logits = anchor_dot_contrast - logits_max.detach()      #每一行减去最大值

        mask = mask.repeat(anchor_count, contrast_count)        #[384,384]，标记了每行来自同一个样本的三个增强，mask[0][0]是自身和自身对比

        # mask-out self-contrast cases
        logits_mask = torch.scatter(                            #[384,384]对角线全是0
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask                               #每行，和自身的另外两个增强样本对比 []
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask            #除了对角线，每个元素取指数

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))      #

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss