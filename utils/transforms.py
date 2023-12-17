import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, liver_imgs, liver_mask=None):
        for t in self.transforms:
            liver_imgs, liver_mask = t(liver_imgs, liver_mask)
        return liver_imgs, liver_mask


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, liver_imgs, liver_mask=None):
        if torch.rand(1) < self.p:
            if isinstance(liver_imgs, list):
                for i in range(len(liver_imgs)):
                    liver_imgs[i] = F.hflip(liver_imgs[i])
            else:
                liver_imgs = F.hflip(liver_imgs)

            if liver_mask is not None:
                if isinstance(liver_mask, list):
                    for i in range(len(liver_mask)):
                        liver_mask[i] = F.hflip(liver_mask[i])
                else:
                    liver_mask = F.hflip(liver_mask)

        return liver_imgs, liver_mask


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, liver_imgs, liver_mask=None):
        if torch.rand(1) < self.p:
            if isinstance(liver_imgs, list):
                for i in range(len(liver_imgs)):
                    liver_imgs[i] = F.vflip(liver_imgs[i])
            else:
                liver_imgs = F.vflip(liver_imgs)

            if liver_mask is not None:
                if isinstance(liver_mask, list):
                    for i in range(len(liver_mask)):
                        liver_mask[i] = F.vflip(liver_mask[i])
                else:
                    liver_mask = F.vflip(liver_mask)

        return liver_imgs, liver_mask


class CenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.crop = T.CenterCrop(size)

    def forward(self, liver_imgs, liver_mask=None):

        if isinstance(liver_imgs, list):
            for i in range(len(liver_imgs)):
                liver_imgs[i] = self.crop(liver_imgs[i])
        else:
            liver_imgs = self.crop(liver_imgs)

        if liver_mask is not None:
            if isinstance(liver_mask, list):
                for i in range(len(liver_mask)):
                    liver_mask[i] = self.crop(liver_mask[i])
            else:
                liver_mask = self.crop(liver_mask)

        return liver_imgs, liver_mask


class ToTensor(object):
    def __call__(self, liver_imgs, liver_mask=None):
        if isinstance(liver_imgs, list):
            for i in range(len(liver_imgs)):
                liver_imgs[i] = F.to_tensor(liver_imgs[i])
        else:
            liver_imgs = F.to_tensor(liver_imgs)

        if liver_mask is not None:
            liver_mask = F.to_tensor(liver_mask)

        return liver_imgs, liver_mask


class Normalize(torch.nn.Module): #z-score standardization
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, liver_imgs, liver_mask=None):
        if isinstance(liver_imgs, list):
            liver_imgs = torch.cat(tuple(liver_imgs))
    
        liver_imgs = F.normalize(liver_imgs.float(), self.mean, self.std)

        return liver_imgs, liver_mask