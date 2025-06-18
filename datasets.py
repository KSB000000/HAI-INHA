import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=False, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            # 테스트셋: 라벨 없이 이미지 경로만 저장
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            # 학습셋: 클래스별 폴더 구조에서 라벨 추출
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                if isinstance(self.transform, transforms.Compose):  # torchvision transforms
                    image = self.transform(image)
                else:  # albumentations
                    image = np.array(image)
                    image = self.transform(image=image)['image']
                # image = self.transform(image=image)['image']
            return image
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            if self.transform:
                if isinstance(self.transform, transforms.Compose):  # torchvision transforms
                    image = self.transform(image)
                else:  # albumentations
                    image = np.array(image)
                    image = self.transform(image=image)['image']
                # image = self.transform(image=image)['image']
            return image, label
