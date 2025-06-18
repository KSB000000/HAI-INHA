from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


# ImageNet normalization
imagenet_mean = [0.4120, 0.4055, 0.4209]
imagenet_std  = [0.3138, 0.3144, 0.3138]


class SquarePadResize:
    """비율 유지하며 padding 추가 후 resize"""
    def __init__(self, size=384):
        self.size = size


    def __call__(self, img: Image.Image):
        w, h = img.size
        max_side = max(w, h)
        hp = (max_side - w) // 2
        vp = (max_side - h) // 2
        padding = (hp, vp, max_side - w - hp, max_side - h - vp)  # left, top, right, bottom
        img = F.pad(img, padding, fill=0, padding_mode='constant')
        img = F.resize(img, (self.size, self.size))
        return img


# RandAugment 기반 강한 augmentation (비율 유지)
def get_train_transform(input_size):
    return transforms.Compose([
        SquarePadResize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


# 검증용 transform (비율 유지)
def get_val_transform(input_size):
    return transforms.Compose([
        SquarePadResize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
