import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False, include_crop=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.include_crop = include_crop
        self.samples = []

        self.group_classes = set([
            "K8_2022_2024", "K8_하이브리드_2022_2024",
            "올_뉴_K7_2016_2019", "A7_2012_2016", "올_뉴_K7_하이브리드_2017_2019",
            "K5_3세대_2020_2023", "K5_3세대_하이브리드_2020_2022", "K5_하이브리드_3세대_2020_2023",
            "K7_프리미어_2020_2021", "K7_프리미어_하이브리드_2020_2021",
            "그랜드_스타렉스_2016_2018", "더_뉴_그랜드_스타렉스_2018_2021",
            "카니발_4세대_2021", "카니발_4세대_2022_2023",
            "4시리즈_G22_2021_2023", "4시리즈_G22_2024_2025", "i4_2022_2024",
            "레인지로버_이보크_2세대_2020_2022", "레인지로버_이보크_2세대_2023_2024",
            "스팅어_2018_2020", "스팅어_마이스터_2021_2023",
            "아반떼_CN7_2021_2023", "아반떼_하이브리드_CN7_2021_2023",
            "티볼리_2015_2018", "티볼리_아머_2018_2019",
            "X6_G06_2020_2023", "X6_G06_2024_2025",
            "XM3_2020_2023", "XM3_2024",
            "EQE_V295_2022_2024", "EQS_V297_2022_2023",
            "레니게이드_2015_2017", "레니게이드_2019_2023",
            "렉스턴_스포츠_2018_2021", "렉스턴_스포츠_칸_2019_2020",
            "718_박스터_2017_2024", "718_카이맨_2017_2024",
            "트레일블레이저_2021_2022", "트레일블레이저_2023",
            "더_넥스트_스파크_2016_2018", "더_뉴_스파크_2019_2022",
            "3008_2세대_2018_2023", "5008_2세대_2018_2019", "5008_2세대_2021_2024",
            "더_뉴_렉스턴_스포츠_2021_2025", "더_뉴_렉스턴_스포츠_칸_2021_2025",
            "뉴_QM6_2021_2023", "더_뉴_QM6_2020_2023",
            "6시리즈_GT_G32_2018_2020", "6시리즈_GT_G32_2021_2024",
            "리얼_뉴_콜로라도_2021_2022", "콜로라도_2020_2020",
            "GLC_클래스_X253_2020_2022", "GLC_클래스_X253_2023",
            "더_뉴_K5_3세대_2024_2025", "더_뉴_K5_하이브리드_3세대_2023_2025"
        ])

        if is_test:
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
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
                    is_crop = '_crop_' in fname

                    if not include_crop and is_crop:
                        continue

                    self.samples.append((img_path, label))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.is_test:
            img_path = sample[0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image

        if len(sample) == 2:
            img_path, label = sample
            image = Image.open(img_path).convert('RGB')
        else:
            img_path, label, direction = sample
            image = Image.open(img_path).convert('RGB')
            image = self.crop_image(image, direction)

        if self.transform:
            image = self.transform(image)
        return image, label


    def crop_image(self, image, direction):
        width, height = image.size
        if direction == 'top':
            return image.crop((0, 0, width, int(height * 2 / 3)))
        elif direction == 'bottom':
            return image.crop((0, int(height * 1 / 3), width, height))
        elif direction == 'left':
            return image.crop((0, 0, int(width * 2 / 3), height))
        elif direction == 'right':
            return image.crop((int(width * 1 / 3), 0, width, height))
        else:
            raise ValueError(f"Unknown direction: {direction}")
