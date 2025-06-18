import albumentations as A
from albumentations.pytorch import ToTensorV2


# Normalization
car_mean = [0.4120, 0.4055, 0.4209]
car_std  = [0.3138, 0.3144, 0.3138]


# Training용 transform
def get_train_transform(image_size):
    return A.Compose([
        # 1. Resize to target size (e.g., 224x224 or 384x384 depending on your backbone)
        A.Resize(height=image_size, width=image_size),

        # 2. Augmentation
        A.OneOf([
            A.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.CenterCrop(image_size, image_size, p=0.5),
        ], p=0.7),

        A.HorizontalFlip(p=0.5),  # Fine-grained에서는 Flip만큼은 적당히 허용
        A.VerticalFlip(p=0.1),    # 자동차 위/아래 Flip은 제한적으로

        # 색상 보정은 약하게 (강하면 정보 왜곡 위험)
        A.OneOf([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05, p=0.7),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        ], p=0.5),

        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.1),
        ], p=0.3),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.CLAHE(clip_limit=2, p=0.2),
        ], p=0.3),

        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            shear={"x": 0.0, "y": 0.0},
            p=0.5
        ),

        # Cutout-like augmentation은 강하게 안 넣는 게 좋아
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 8),
            hole_width_range=(8, 8),
            fill=0,
            p=0.2
        ),

        # Normalize: ImageNet mean & std
        A.Normalize(mean=car_mean, std=car_std),

        ToTensorV2(),
    ])


# Validation용 transform
def get_val_transform(input_size):
    return A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=car_mean, std=car_std),
        ToTensorV2(),
    ])


def get_test_transform(input_size):
    import cv2
    return A.Compose([
        A.LongestMaxSize(max_size=input_size),

        A.PadIfNeeded(
            min_height=input_size,
            min_width=input_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),  # RGB 이미지용 padding 색상
            position='center'
        ),

        A.Normalize(mean=car_mean, std=car_std),
        ToTensorV2(),
    ])
