import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataloader.transform import PretrainTransform
from dataloader.transform import PretrainTransform_crop_big_single
# from dataloader.constrained_cropping import *

#train:50000张图片，100个类，每个类500张图片。  test:10000张图片，100个类，每个类100张图片
miniImageNet_classes =["house finch","American robin","triceratops","green mamba","harvestman","toucan","goose","jellyfish","nematode","red king crab",
                        "dugong","Treeing Walker Coonhound","Ibizan Hound","Saluki","Golden Retriever","Gordon Setter","Komondor","Boxer","Tibetan Mastiff","French Bulldog",
                        "Alaskan Malamute","Dalmatian","Newfoundland dog","Miniature Poodle","Alaskan tundra wolf","African wild dog","Arctic fox","lion","meerkat","ladybug",
                        "rhinoceros beetle","ant","black-footed ferret","three-toed sloth","rock beauty fish","aircraft carrier","trash can","barrel","beer bottle","bookstore",
                        "cannon","carousel","cardboard box / carton","catamaran","bell or wind chime","clogs","cocktail shaker","combination lock","crate","cuirass",
                        "dishcloth","dome","electric guitar","filing cabinet","fire screen","frying pan","garbage truck","hair clip","holster","gymnastic horizontal bar",
                        "hourglass","iPod","lipstick","miniskirt","missile","mixing bowl","oboe","pipe organ","parallel bars","pencil case",
                        "photocopier","poncho","prayer rug","fishing casting reel","school bus","scoreboard","slot machine","snorkel","solar thermal collector","spider web",
                        "stage","tank","front curtain","tile roof","tobacco shop","unicycle","upright piano","vase","wok","split-rail fence",
                        "sailboat","traffic or street sign","consomme","trifle","hot dog","orange","cliff","coral reef","bolete","corn cob"
]

#clip_use
imagenet_templates = ["itap of a {}.",
                        "a bad photo of the {}.",
                        "a photo of a large {}.",
                        "a {} in a video game.",
                        "art of the {}.",
                        "a photo of a small {}.",
                        'a photo of many {}.',
                        'a sculpture of a {}.',
                        'a photo of the hard to see {}.',
'a low resolution photo of the {}.',
'a cropped photo of the {}.',
'a photo of a hard to see {}.',
'a bright photo of a {}.',
'a photo of a clean {}.',
'a dark photo of the {}.',
'a drawing of a {}.',
'a close-up photo of a {}.',
'a pixelated photo of the {}.',
'a bright photo of the {}.',
'a cropped photo of a {}.',
'a blurry photo of the {}.',
'a photo of the {}.',
'a good photo of the {}.',
'a photo of one {}.',
'a close-up photo of the {}.',
'a photo of a {}.',
'the {} in a video game.',
'a low resolution photo of a {}.',
'a rendition of the {}.',
'a photo of the clean {}.',
'a photo of a nice {}.',
'a blurry photo of a {}.',
'a pixelated photo of a {}.',
'a good photo of a {}.',
'a photo of the nice {}.',
'a photo of the small {}.',
'a dark photo of a {}.',
'a photo of a cool {}.',]

class ConvertToRGB(object):
    def __call__(self, image):
        """  
        将输入图像转换为 RGB 格式  
        """  
        if image.mode != 'RGB':  
            image = image.convert('RGB')  
        return image

class MiniImageNet(Dataset):

    def __init__(self, root='./data', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None, args=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet/split')

        self.return_idx = False
        
        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb
        if train:
        #if False:
            image_size = 84
            self.transform = PretrainTransform('mini_imagenet', args)
            # self.transform_tmp = PretrainTransform('mini_imagenet', args)
            # self.transform_key = PretrainTransform_crop_big_single('mini_imagenet', args)
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([96, 96]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        
    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        original_img = Image.open(path).convert('RGB')
        image = self.transform( original_img)   #返回原始图像的num_aug个增强样本
        
        if self.return_idx:
            return image,targets,i
        
        return image, targets


