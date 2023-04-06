import torch
from torchvision.transforms import autoaugment, transforms
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

def create_augment(augmentation, mean, std):
    aug = [T.Lambda(lambda x: x.convert("RGB"))]
    for i in augmentation.split("-"):
        if "identity" in i:
           aug.extend([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),])
        elif "crop" in i:
            aug.append(T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC)  ) 
        elif "hflip" in i:
            _, prob = i.split("_")
            aug.append(T.RandomHorizontalFlip(p=float(prob)))
        elif "randaug" in i:
            _, n, m, prob = i.split("_")
            aug.append( T.RandomApply(transforms=[
                           T.RandAugment(num_ops=int(n[1:]), magnitude=int(m[1:]), interpolation=T.InterpolationMode.BICUBIC)
                        ], p=float(prob[1:])) )
    aug.extend([transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float), T.Normalize(mean, std)])

    return T.Compose(aug)

class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        first_transform,
        second_transform,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        same_crop=True,
        interpolation=InterpolationMode.BICUBIC,
        tied_augment=False,
    ):
        self.resize = transforms.RandomResizedCrop(crop_size, interpolation=interpolation)
        self.tied_augment = tied_augment
        self.same_crop = same_crop
        if not tied_augment:
            self.transform = create_augment(first_transform, mean, std)
        else:
            self.first_transform = create_augment(first_transform.replace("crop-", ""), mean, std)
            self.second_transform = create_augment(second_transform.replace("crop-", ""), mean, std)

    def __call__(self, img):
        if not self.tied_augment:
          return self.transform(self.resize(img))
        if self.same_crop:
            img = self.resize(img)
            first, second = img, img.copy()
        else:
            first, second = self.resize(img), self.resize(img)
        
        return self.first_transform(first), self.second_transform(second)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BICUBIC,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)
