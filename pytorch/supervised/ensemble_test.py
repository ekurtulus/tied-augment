from models import WideResNet
import torch
from torch import nn
import numpy as np
import os
import glob
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd 
from augmentations import Cutout

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        logits = self.fc(out)
        return out, logits

            

identity_set = DataLoader(datasets.CIFAR10(root="../datasets", train=False, 
                                       transform=T.Compose([T.ToTensor(), T.Normalize(
                                                                mean=torch.tensor([0.4914, 0.4822, 0.4465]),
                                                                std=torch.tensor([0.2475, 0.2435, 0.2615]))
                                                           ])
                                      ), batch_size=4096, num_workers=16, shuffle=False)

randaug_set = DataLoader(datasets.CIFAR10(root="../datasets", train=False, 
                                       transform=T.Compose([T.RandAugment(num_ops=2, magnitude=14), T.ToTensor(), 
                                                            T.Normalize(
                                                                mean=torch.tensor([0.4914, 0.4822, 0.4465]),
                                                                std=torch.tensor([0.2475, 0.2435, 0.2615])),
                                                           ])
                                      ), batch_size=4096, num_workers=16, shuffle=False)

hflip_set = DataLoader(datasets.CIFAR10(root="../datasets", train=False, 
                                       transform=T.Compose([T.RandomHorizontalFlip(p=1), T.ToTensor(), 
                                                            T.Normalize(
                                                                mean=torch.tensor([0.4914, 0.4822, 0.4465]),
                                                                std=torch.tensor([0.2475, 0.2435, 0.2615])),
                                                           ])
                                      ), batch_size=4096, num_workers=16, shuffle=False)

vflip_set = DataLoader(datasets.CIFAR10(root="../datasets", train=False, 
                                       transform=T.Compose([T.RandomVerticalFlip(p=1), T.ToTensor(), 
                                                            T.Normalize(
                                                                mean=torch.tensor([0.4914, 0.4822, 0.4465]),
                                                                std=torch.tensor([0.2475, 0.2435, 0.2615])),
                                                           ])
                                      ), batch_size=4096, num_workers=16, shuffle=False)

full_set = DataLoader(datasets.CIFAR10(root="../datasets", train=False, 
                                       transform=T.Compose([T.RandomHorizontalFlip(p=0.5), 
                                                            T.RandAugment(num_ops=2, magnitude=14),
                                                            T.ToTensor(), 
                                                            T.Normalize(
                                                                mean=torch.tensor([0.4914, 0.4822, 0.4465]),
                                                                std=torch.tensor([0.2475, 0.2435, 0.2615])),
                                                           Cutout()],
                                                          )
                                      ), batch_size=4096, num_workers=16, shuffle=False)

res = {}

for folder in tqdm([i for i in glob.glob("experiments/from_scratch_cifar10/wrn28-10*") 
                    if os.path.isfile(os.path.join(i, "run_0", "final.pt"))]):
    models = [WideResNet(28, 10, 10).eval().cuda() for i in range(5)]
    for i in range(5):
        models[i].load_state_dict(torch.load(os.path.join(folder, "run_" + str(i), "final.pt"))["model"])

    temp = {}
    
    for test_set, name in zip([identity_set,
                                   randaug_set,
                                   hflip_set,
                                   vflip_set,
                                   full_set,],
                                  ["identity",
                                   "randaug",
                                   "hflip",
                                   "vflip",
                                   "hflip_randaug",]):
        features, logits = [], []
        for x,y in test_set:
            x = x.cuda()
            temp_features, temp_logits = [], []
            with torch.no_grad():
                for i in range(5):
                    feat, logit = models[i](x)
                    logit = torch.softmax(logit, -1)
                    temp_features.append(feat.cpu())
                    temp_logits.append(logit.cpu())
            
            features.append(temp_features)
            logits.append(temp_logits)
        
        logits_ = torch.stack([torch.cat([i[j] for i in logits], 0) for j in range(5)])
        features_ = torch.stack([torch.cat([i[j] for i in features], 0) for j in range(5)])
        
        temp[name + "_logits"] = logits_.numpy()
        temp[name + "_features"] = features_.numpy()
     
    np.savez("features/cifar10/wrn28-10/" + os.path.basename(folder) + ".npz", **temp)
   

for folder in tqdm([i for i in glob.glob("experiments/from_scratch_cifar10/wrn28-2*") 
                    if os.path.isfile(os.path.join(i, "run_0", "final.pt"))  ]):
    model = WideResNet(28, 10, 2).eval().cuda()
    model.load_state_dict(torch.load(os.path.join(folder, "run_0", "final.pt"))["model"])

    temp = {}
    
    for test_set, name in zip([identity_set,
                                   randaug_set,
                                   hflip_set,
                                   vflip_set,
                                   full_set,],
                                  ["identity",
                                   "randaug",
                                   "hflip",
                                   "vflip",
                                   "hflip_randaug",]):
        features, logits = [], []
        for x,y in test_set:
            x = x.cuda()
            temp_features, temp_logits = [], []
            with torch.no_grad():
                for i in range(5):
                    feat, logit = models[i](x)
                    logit = torch.softmax(logit, -1)
                    temp_features.append(feat.cpu())
                    temp_logits.append(logit.cpu())
            
            features.append(temp_features)
            logits.append(temp_logits)
        
        logits_ = torch.stack([torch.cat([i[j] for i in logits], 0) for j in range(5)])
        features_ = torch.stack([torch.cat([i[j] for i in features], 0) for j in range(5)])
        
        temp[name + "_logits"] = logits_.numpy()
        temp[name + "_features"] = features_.numpy()
     
    np.savez("features/cifar10/wrn28-2/" + os.path.basename(folder) + ".npz", **temp)

    

