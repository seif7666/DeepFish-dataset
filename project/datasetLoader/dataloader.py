from __future__ import print_function, division
import sys
import os
from typing import Any
import torch
import numpy as np
import random
import csv
import cv2
from time import time
import skimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
from PIL import Image
from torch.utils.data import Dataset,DataLoader
try:
    from .EstimationLoader import EstimationLoader
except:
    from EstimationLoader import EstimationLoader
import torch

from torchvision.transforms import Compose,ToTensor,Resize
import random 


def is_tensor(x)->bool:
    return type(x)==torch.Tensor or type(x)==np.ndarray

class EstimatedDeepFish(Dataset):
    def __init__(self, csv_path:str, dataset_path:str, transform=None, for_train=True) -> None:
        super().__init__()
        self.__estimationLoader= EstimationLoader(csv_path,dataset_path,for_train)
        self.transform= transform
        self.estimated_time= 0

        

    def __getitem__(self, idx) -> dict:
        # current= time()
        dictionary= self.__estimationLoader[idx]
        # img = self.load_image(idx)
        # annot = self.load_annotations(idx)
        sample = {'img': dictionary['image'], 'annot': dictionary['annots']}
        if self.transform:
            sample = self.transform(sample)
        
        img= sample['img']
        if img.shape[1]!=384 or img.shape[2]!=512:
            return self[random.randint(0,len(self.__estimationLoader))]
        sample['number']= dictionary['number']
        # self.estimated_time+=(time()-current)*1000
        # print(f'Time taken is {time()-current} seconds')
        return sample
    
    def num_classes(self):
        return self.__estimationLoader.getNumClasses()
    def __len__(self):
        return len(self.__estimationLoader)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,min_side=608, max_side= 1024):
        super().__init__()
        self.minSide= min_side
        self.maxSide= max_side

    def __call_for_tensor(self,sample):
        min_side,max_side= self.minSide,self.maxSide
        image= sample
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        image= cv2.resize(image,(int(round((cols*scale))),int(round(rows*scale))))
        rows, cols, cns = image.shape
        pad_w = 32 - rows%32
        pad_h = 32 - cols%32
        new_image = torch.zeros((rows + pad_w, cols + pad_h, cns))
        new_image[:rows, :cols, :] = torch.from_numpy(image)
        return new_image

    def __call__(self, sample):
        if is_tensor(sample):
            return self.__call_for_tensor(sample)
        min_side,max_side= self.minSide,self.maxSide
        image, annots = sample['img'], sample['annot']
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        image= cv2.resize(image,(int(round((cols*scale))),int(round(rows*scale))))
        rows, cols, cns = image.shape
        pad_w = 32 - rows%32
        pad_h = 32 - cols%32
        new_image = torch.zeros((rows + pad_w, cols + pad_h, cns))
        new_image[:rows, :cols, :] = torch.from_numpy(image)
        annots[:, :4] *= scale
        return {'img': new_image, 'annot': annots, 'scale': scale}

class Permuter(object):
    def __call__(self, sample):
        if is_tensor(sample):
            return sample.permute((2,0,1))
        sample['img']=sample['img'].permute((2,0,1))
        return sample
class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]
            rows, cols, channels = image.shape

            x1 = annots[:, 0].clone()
            x2 = annots[:, 2].clone()
            
            x_tmp = x1.clone()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample['img']= image
            sample['annots']=annots

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = torch.tensor([[[0.485, 0.456, 0.406]]])
        self.std = torch.tensor([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        if is_tensor(sample):
            # sample= (sample-self.mean)/self.std
            return sample.numpy()
        # sample['img]']=(sample['img']-self.mean)/self.std
        sample['img']= sample['img'].numpy()
        return sample

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]



def load_image(path):
    modelTransform= transforms.Compose([Normalizer(), Resizer(480,480),Permuter()])
    visionTransform= transforms.Compose([Resizer(480,480)])
    img = skimage.io.imread(path)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)
    return modelTransform(torch.from_numpy(img)/255.0), visionTransform(img)


if __name__=='__main__':
    dataset= EstimatedDeepFish('Project/size_estimation_homography_DeepFish.csv', 'Project/DATASET/',Compose([Normalizer(), Augmenter(), Resizer(480,480),Permuter()]))
    # print(dataset[0])
    dataloader= DataLoader(dataset,8)
    next(iter(dataloader))
    print(f'Average time is {dataset.estimated_time/8}ms')
