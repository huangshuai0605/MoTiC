import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader.transform import PretrainTransform
from torchvision.transforms import InterpolationMode

cub200_classes = ["Black_footed_Albatross","Laysan_Albatross","Sooty_Albatross","Groove_billed_Ani","Crested_Auklet","Least_Auklet","Parakeet_Auklet","Rhinoceros_Auklet","Brewer_Blackbird","Red_winged_Blackbird",
                  "Rusty_Blackbird","Yellow_headed_Blackbird","Bobolink","Indigo_Bunting","Lazuli_Bunting","Painted_Bunting","Cardinal","Spotted_Catbird","Gray_Catbird","Yellow_breasted_Chat",
                  "Eastern_Towhee","Chuck_will_Widow","Brandt_Cormorant","Red_faced_Cormorant","Pelagic_Cormorant","Bronzed_Cowbird","Shiny_Cowbird","Brown_Creeper","American_Crow","Fish_Crow",
                  "Black_billed_Cuckoo","Mangrove_Cuckoo","Yellow_billed_Cuckoo","Gray_crowned_Rosy_Finch","Purple_Finch","Northern_Flicker","Acadian_Flycatcher","Great_Crested_Flycatcher","Least_Flycatcher","Olive_sided_Flycatcher",
                  "Scissor_tailed_Flycatcher","Vermilion_Flycatcher","Yellow_bellied_Flycatcher","Frigatebird","Northern_Fulmar","Gadwall","American_Goldfinch","European_Goldfinch","Boat_tailed_Grackle","Eared_Grebe",
                  "Horned_Grebe","Pied_billed_Grebe","Western_Grebe","Blue_Grosbeak","Evening_Grosbeak","Pine_Grosbeak","Rose_breasted_Grosbeak","Pigeon_Guillemot","California_Gull","Glaucous_winged_Gull",
                  "Heermann_Gull","Herring_Gull","Ivory_Gull","Ring_billed_Gull","Slaty_backed_Gull","Western_Gull","Anna_Hummingbird","Ruby_throated_Hummingbird","Rufous_Hummingbird","Green_Violetear",
                  "Long_tailed_Jaeger","Pomarine_Jaeger","Blue_Jay","Florida_Jay","Green_Jay","Dark_eyed_Junco","Tropical_Kingbird","Gray_Kingbird","Belted_Kingfisher","Green_Kingfisher",
                  "Pied_Kingfisher","Ringed_Kingfisher","White_breasted_Kingfisher","Red_legged_Kittiwake","Horned_Lark","Pacific_Loon","Mallard","Western_Meadowlark","Hooded_Merganser","Red_breasted_Merganser",
                  "Mockingbird","Nighthawk","Clark_Nutcracker","White_breasted_Nuthatch","Baltimore_Oriole","Hooded_Oriole","Orchard_Oriole","Scott_Oriole","Ovenbird","Brown_Pelican"

                  "White Pelican","Western Wood Pewee","Sayornis","American Pipit","Whip-poor-Will","Horned Puffin","Common Raven","White-necked Raven","American Redstart","Geococcyx"
                  "Loggerhead Shrike","Great Grey Shrike","Baird Sparrow","Black-throated Sparrow","Brewer Sparrow","Chipping Sparrow","Clay-colored Sparrow","House Sparrow","Field Sparrow","Fox Sparrow",
                  "Grasshopper Sparrow","Harris Sparrow","Henslow Sparrow","Le Conte Sparrow","Lincoln Sparrow","Nelson Sharp_tailed Sparrow","Savannah Sparrow","Seaside Sparrow","Song Sparrow","Tree Sparrow",
                  "Vesper Sparrow","White_crowned Sparrow","White_throated Sparrow","Cape_Glossy Starling","Bank Swallow","Barn Swallow","Cliff Swallow","Tree Swallow","Scarlet Tanager","Summer Tanager",
                  "Artic Tern","Black Tern","Caspian Tern","Common Tern","Elegant Tern","Forsters Tern","Least Tern","Green_tailed Towhee","Brown Thrasher","Sage Thrasher",
                  "Black_capped Vireo","Blue_headed Vireo","Philadelphia Vireo","Red_eyed Vireo","Warbling Vireo","White_eyed Vireo","Yellow_throated Vireo","Bay_breasted Warbler","Black_and_white Warbler","Black_throated_Blue Warbler",
                  "Blue_winged_Warbler","Canada_Warbler","Cape_May_Warbler","Cerulean_Warbler","Chestnut_sided_Warbler","Golden_winged_Warbler","Hooded_Warbler","Kentucky_Warbler","Magnolia_Warbler","Mourning_Warbler",
                  "Myrtle_Warbler","Nashville_Warbler","Orange_crowned_Warbler","Palm_Warbler","Pine_Warbler","Prairie_Warbler","Prothonotary_Warbler","Swainson_Warbler","Tennessee_Warbler","Wilson_Warbler",
                  "Worm_eating_Warbler","Yellow_Warbler","Northern_Waterthrush","Louisiana_Waterthrush","Bohemian_Waxwing","Cedar_Waxwing","American_Three_toed_Woodpecker","Pileated_Woodpecker","Red_bellied_Woodpecker","Red_cockaded_Woodpecker",
                  "Red_headed_Woodpecker","Downy_Woodpecker","Bewick_Wren","Cactus_Wren","Carolina_Wren","House_Wren","Marsh_Wren","Rock_Wren","Winter_Wren","Common_Yellowthroat"
]

#clip_use
cub200_templates = [  
    "a photo of a {} bird.",  
    "a stunning image of a {}.",  
    "a portrait of a {} bird.",  
    "a close-up of a {}.",  
    "a photo of a colorful {}.",  
    "an illustration of a {} bird.",  
    "a beautiful photo of a {}.",  
    "a dark photo of a {} bird.",  
    "a high-resolution photo of a {}.",  
    "a long shot of a {} bird.",  
    "a photo of multiple {} birds.",  
    "a clear image of a {}.",  
    "a detailed view of a {} bird.",  
    "a profile shot of a {}.",  
    "a photo of a {} in flight.",  
    "a whimsical drawing of a {} bird.",  
    "a blurred photo of a {}.",  
    "a photo showcasing the feathers of a {}.",  
    "a natural habitat photo of the {}.",  
    "a low angle shot of a {} bird.",  
    "a photo capturing a {} in its environment.",  
    "a majestic photo of a {} bird.",  
    "a artistic rendition of a {}.",  
    "a photo of a baby {}.",  
    "a photo of a rare {} bird.",  
    "the {} bird perched on a branch.",  
    "a birdwatching photo of a {}.",  
    "a cheerful photo of a {} bird.",  
    "a nighttime photo of a {} bird.",  
    "a side view of a {}.",  
    "a close-up of the head of a {} bird.",  
    "a photo of a nest with a {}."  
]  

class ConvertToRGB(object):
    def __call__(self, image):
        """  
        将输入图像转换为 RGB 格式  
        """  
        if image.mode != 'RGB':  
            image = image.convert('RGB')  
        return image

#clip_vit 16 use preprocess
clip_transform = transforms.Compose([
    transforms.Resize( size=(224,224),interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop( size=(224,224)),
    ConvertToRGB(),
    transforms.ToTensor(),
    transforms.Normalize( mean=[0.48145466,0.4578275,0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

#base session:3000 pictures, each class 30张图片，总共100个类
class CUB200(Dataset):

    def __init__(self, root='./', train=True,
                 index_path=None, index=None, base_sess=None, args=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self._pre_operate(self.root)

        # self.clip = False
        # self.clip_transform = clip_transform
        self.return_idx = False
        
        if train:
        #if False:
            self.transform = PretrainTransform('cub200', args)
            # self.transform_tmp = PretrainTransform('cub200', args)
            # self.transform_FSCIL = transforms.Compose([
            #     transforms.RandomResizedCrop(224, scale=(0.25,1), ratio=(1,1)),
            #     transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),  
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
            
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
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
        image = self.transform(Image.open(path).convert('RGB'))
        
        if self.return_idx:
            return image,targets,i
        # if self.clip:
        #     clip_image = self.clip_transform(Image.open(path) )
        #     return image, targets,clip_image
        
        return image, targets

