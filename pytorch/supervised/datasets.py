import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T
from functools import partial
from augmentations import *
from math import ceil

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

def reduced_cifar(root, cifar_type, num_labels, train=False, transform=None):
    if not train:
        return datasets.CIFAR10(root, train=False, transform=transform) if cifar_type == "cifar10" else \
                datasets.CIFAR100(root, train=False, transform=transform)
        
    data = datasets.CIFAR10(root, transform=transform) if cifar_type == "cifar10" else datasets.CIFAR100(root, transform=transform)
    per_class = num_labels // 10 if cifar_type == "cifar10" else num_labels // 100
    
    images, targets = data.data, data.targets
    
    permutation_tensor = torch.randperm(50_000)
    images, targets = images[permutation_tensor], torch.tensor(targets)[permutation_tensor].numpy().tolist()
    
    sampled_images, sampled_targets = [], []
    
    sampled_count = [0] * 10 if cifar_type == "cifar10" else [0] * 100
    
    for j, i in enumerate(targets):
        if sampled_count[i] < per_class:
            sampled_images.append( images[j] )
            sampled_targets.append( targets[j] )
            sampled_count[i] += 1
    
    data.data = sampled_images
    data.targets = sampled_targets
    
    return data
    

DATASETS = {
    "cifar10" : {
        "dataset_function" : datasets.CIFAR10,
        "mean" : torch.tensor([0.4914, 0.4822, 0.4465]),
        "std" : torch.tensor([0.2475, 0.2435, 0.2615]),
        "num_classes" : 10,
        "split_args" : {"train" : {"train" : True}, "test" : {"train" : False}},
        "evaluation_strategy" : "top-1"
        #"evaluation_strategy" : "per-class-mean"
    },
    "cifar100" : {
        "dataset_function" : datasets.CIFAR100,
        "mean" : torch.tensor([0.5070, 0.4865, 0.4409]),
        "std" : torch.tensor([0.2673, 0.2564, 0.2761]),
        "num_classes" : 100,
        "split_args" : {"train" : {"train" : True}, "test" : {"train" : False}},
        "evaluation_strategy" : "top-1"
    },
    "reduced_cifar10" : {
        "dataset_function" : partial(reduced_cifar, cifar_type="cifar10", num_labels=4000),
        "mean" : torch.tensor([0.4914, 0.4822, 0.4465]),
        "std" : torch.tensor([0.2475, 0.2435, 0.2615]),
        "num_classes" : 10,
        "split_args" : {"train" : {"train" : True}, "test" : {"train" : False}},
        "evaluation_strategy" : "top-1"
        #"evaluation_strategy" : "per-class-mean"
    },
    "reduced_cifar100" : {
        "dataset_function" : partial(reduced_cifar, cifar_type="cifar100", num_labels=10000),
        "mean" : torch.tensor([0.5070, 0.4865, 0.4409]),
        "std" : torch.tensor([0.2673, 0.2564, 0.2761]),
        "num_classes" : 100,
        "split_args" : {"train" : {"train" : True}, "test" : {"train" : False}},
        "evaluation_strategy" : "top-1"
    },
    "svhn" : {
        "dataset_function" : datasets.SVHN,
        "mean" : torch.tensor([0.4309, 0.4302, 0.4463]),
        "std" : torch.tensor([0.1975, 0.2002, 0.1981]),
        "num_classes" : 10,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "test"}},
        "evaluation_strategy" : "top-1"
    },
    "aircraft-variant" : {
        "dataset_function" : partial(datasets.FGVCAircraft, annotation_level="variant"),
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 100,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "val"}},
        "evaluation_strategy" : "per-class-mean"
    },
    "aircraft-manufacturer" : {
        "dataset_function" : partial(datasets.FGVCAircraft, annotation_level="variant"),
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 30,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "val"}},
        "evaluation_strategy" : "per-class-mean"
    },
    "aircraft-family" : {
        "dataset_function" : partial(datasets.FGVCAircraft, annotation_level="variant"),
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 70,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "val"}},
        "evaluation_strategy" : "per-class-mean"
    },
    "flowers102" : {
        "dataset_function" : datasets.Flowers102,
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 102,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "val"}},
        "evaluation_strategy" : "per-class-mean"
    },
    "food101" : {
        "dataset_function" : datasets.Food101,
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 101,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "test"}},
        "evaluation_strategy" : "top-1"
    },
    "stanford-cars" : {
        "dataset_function" : datasets.StanfordCars,
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 196,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "test"}},
        "evaluation_strategy" : "top-1"
    },
    "oxford-pets" : {
        "dataset_function" : datasets.OxfordIIITPet,
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 37,
        "split_args" : {"train" : {"split" : "trainval"}, "test" : {"split" : "test"}},
        "evaluation_strategy" : "per-class-mean"
    },
    "country211": {
        "dataset_function" : datasets.Country211,
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 211,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "valid"}},
        "evaluation_strategy" : "top-1"
    },
    "gtsrb": {
        "dataset_function" : datasets.GTSRB,
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 43,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "test"}},
        "evaluation_strategy" : "top-1"
    },
    "dtd": {
        "dataset_function" : datasets.DTD,
        "mean" : _IMAGENET_MEAN,
        "std" : _IMAGENET_STD,
        "num_classes" : 47,
        "split_args" : {"train" : {"split" : "train"}, "test" : {"split" : "val"}},
        "evaluation_strategy" : "top-1"
    },    
}

class BasicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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

class TwoBranchDataset(Dataset):
    def __init__(self, dataset, first_transform, second_transform, use_same_crop=False):
        self.dataset = dataset
        self.use_same_crop = use_same_crop
        
        if self.use_same_crop:
            self.cropper = _remove_crop_hflip(first_transform)
            _remove_crop_hflip(second_transform)
        self.first_transform = first_transform
        self.second_transform = second_transform

    def __getitem__(self, idx):
        x,y = self.dataset[idx]
        if self.use_same_crop:
            x = self.cropper(x)
        return self.first_transform(x), self.second_transform(x), y

    def __len__(self):
        return len(self.dataset)




def load_dataset(args):
    dataset_args = DATASETS[args.task]
    dataset_function, mean, std, num_classes, split_args, evaluation_strategy = [dataset_args[i] for i in ["dataset_function", "mean", "std", 
                                                                                                            "num_classes", "split_args", "evaluation_strategy"]  ]    
    args.__dict__["num_classes"] = num_classes
    args.__dict__["mean"] = mean
    args.__dict__["std"] = std
    args.__dict__["evaluation_strategy"] = evaluation_strategy

    first_transform = _normalization_handler(torch.load(args.first_augmentation), mean, std)
    second_transform = _normalization_handler(torch.load(args.second_augmentation), mean, std)
    
    if args.model == "wide-resnet":
        test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    else:
        test_transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
        ])
    
    clean_train_data = dataset_function(root=args.datapath, **split_args["train"], transform=test_transform) 
    train_data = dataset_function(root=args.datapath, **split_args["train"], transform=None)
    
    args.__dict__["iterations"] = ceil((len(train_data) * args.epochs) // args.batch_size) if args.num_iters == -1 else args.num_iters
    train_data = TwoBranchDataset(train_data, first_transform, second_transform, use_same_crop=args.use_same_crop)
    
    test_data = dataset_function(root=args.datapath, **split_args["test"], transform=test_transform)
    
    clean_train_set = DataLoader(clean_train_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                           prefetch_factor=args.prefetch_factor, drop_last=False)
    
    train_set = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                           prefetch_factor=args.prefetch_factor, drop_last=True)
    
    test_set = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                           prefetch_factor=args.prefetch_factor, drop_last=False)
    
    
    print("first transform : ", first_transform)
    print("second transform : ", second_transform)
    print("test transform : ", test_transform)
    
    args.__dict__["epoch_every"] = len(train_set)

    return train_set, test_set, clean_train_set
