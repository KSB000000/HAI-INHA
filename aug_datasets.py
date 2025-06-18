import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path, 'original'))  # 기본적으로 원본
        else:
            self.classes = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                label = self.class_to_idx[cls_name]

                for fname in os.listdir(cls_folder):
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    img_path = os.path.join(cls_folder, fname)
                    self.samples.append((img_path, label))  # 원본만 저장


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.is_test:
            img_path, mode = sample
            image = Image.open(img_path).convert('RGB')
        else:
            img_path, label, mode = sample
            image = Image.open(img_path).convert('RGB')

        # 데이터 증강 모드
        if mode == 'crop':
            from random import choice
            ratio = choice([0.5, 2/3, 2/5, 3/5, 4/5])
            image = self.crop_image(image, 'top', ratio)
        elif mode == 'rotate':
            from random import choice
            angle = choice([45, -45, 90, -90])
            image = image.rotate(angle, expand=True)

        if self.transform:
            image = self.transform(image)

        return image if self.is_test else (image, label)


    def crop_image(self, image, direction, ratio=0.5):
        width, height = image.size
        if direction == 'top':
            return image.crop((0, 0, width, int(height * ratio)))
        elif direction == 'bottom':
            return image.crop((0, int(height * (1 - ratio)), width, height))
        elif direction == 'left':
            return image.crop((0, 0, int(width * ratio), height))
        elif direction == 'right':
            return image.crop((int(width * (1 - ratio)), 0, width, height))
        else:
            raise ValueError(f"Unknown direction: {direction}")
