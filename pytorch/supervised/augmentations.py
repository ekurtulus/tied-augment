import numpy as np
import torch
import torchvision.transforms as TF
import os
import math
import random
from torchvision import transforms as T

def create_augment(augmentation, mean, std, finetune=False):
    aug = [T.Lambda(lambda x: x.convert("RGB"))]
    cutout_params = (False, None)
    for i in augmentation.split("-"):
        if "identity" in i:
            if finetune:
                aug.extend([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),])
        elif "crop" in i:
            aug.append(T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC) if finetune else T.RandomCrop(32, padding=4) ) 
        elif "hflip" in i:
            _, prob = i.split("_")
            aug.append(T.RandomHorizontalFlip(p=float(prob)))
        elif "randaug" in i:
            _, n, m, prob = i.split("_")
            aug.append( T.RandomApply(transforms=[
                           T.RandAugment(num_ops=int(n), magnitude=int(m), interpolation=T.InterpolationMode.BICUBIC)
                        ], p=float(prob)) )
        elif "cutout" in i:
            _, c = i.split("_")
            cutout_params = (True, int(c))
    aug.extend([T.ToTensor(), T.Normalize(mean, std)])
    if cutout_params[0]:
        aug.append(Cutout(length=cutout_params[1]))
    
    return T.Compose(aug)

def _remove_crop_hflip(transform):
    temp = []
    for i in range(len(transform.transforms)):
        if isinstance(transform.transforms[i], T.RandomHorizontalFlip) or isinstance(transform.transforms[i], T.RandomCrop):
            temp.append(transform.transforms[i])
            transform.transforms[i] = torch.nn.Identity()   
    return T.Compose(temp)

def _normalization_handler(transform, mean, std):
    for i in range(len(transform.transforms)):
        if isinstance(transform.transforms[i], T.Normalize):
            transform.transforms[i].mean = mean
            transform.transforms[i].std = std
    return transform


class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img
    
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += "n_holes={n_holes}"
        s += ", length={length}"
        s += ")"
        return s.format(**self.__dict__)        
