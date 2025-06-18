import argparse
import os
import json
import numpy as np
import torch
from reg_datasets import CustomImageDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from timm import create_model
from transforms import get_train_transform, get_val_transform
from collate import cutmix_collate_fn
from train import train_model
from calibration import tune_temperature
from seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=bool, default=True)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_root', type=str, default="./data/train")
    parser.add_argument('--save_dir', type=str, default="./checkpoint")
    parser.add_argument('--metrics_file', type=str, default="val_reg.json")
    return parser.parse_args()


def main(args):
    SEED = 42
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 전체 원본 dataset 로드
    full_dataset_no_crop = CustomImageDataset(args.train_root, transform=None, include_crop=False)
    targets = [sample[1] for sample in full_dataset_no_crop.samples]
    class_names = full_dataset_no_crop.classes
    num_classes = len(class_names)
    print(f"총 원본 이미지 수: {len(full_dataset_no_crop)}")

    # Stratified KFold 분할
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=SEED)

    # 모델들 정의
    model_names = [
        "regnety_1280.swag_ft_in1k"
    ]

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n=== Fold {fold+1}/{args.num_folds} ===")

        # ✅ val 데이터셋 (크롭 없음)
        val_samples = [full_dataset_no_crop.samples[i] for i in val_idx]
        val_dataset = CustomImageDataset(args.train_root, transform=get_val_transform(), include_crop=False)
        val_dataset.samples = val_samples

        # ✅ train 데이터셋 (크롭 확장)
        train_samples = [full_dataset_no_crop.samples[i] for i in train_idx]
        train_samples_aug = []

        for img_path, label in train_samples:
            train_samples_aug.append((img_path, label))
            cls_name = full_dataset_no_crop.classes[label]
            if cls_name in full_dataset_no_crop.group_classes:
                for direction in ['top', 'bottom', 'left', 'right']:
                    train_samples_aug.append((img_path, label, direction))

        train_dataset = CustomImageDataset(args.train_root, transform=get_train_transform(), include_crop=False)
        train_dataset.samples = train_samples_aug
        
        print(f"Train 샘플 수 (크롭 포함): {len(train_dataset)}")
        print(f"Val 샘플 수 (크롭 제외): {len(val_dataset)}")


        # Dataloader 구성
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=cutmix_collate_fn,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        for model_name in model_names:
            print(f"▶ Training model: {model_name} (Fold {fold})")

            model = create_model(model_name, pretrained=True, num_classes=num_classes)
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
                model = torch.nn.DataParallel(model)
            model = model.to(device)

            train_model(
                model,
                train_loader,
                val_loader,
                model_name,
                fold,
                num_classes,
                device,
                epochs=args.epochs,
                lr=args.lr,
                save_dir=args.save_dir,
                metrics_file=args.metrics_file
            )

            # ✅ Calibration
            if args.calibration:
                print(f"Calibrating model: {model_name} (Fold {fold})")
                model_path = os.path.join(args.save_dir, f"{model_name}_fold{fold}.pth")

                model_raw = create_model(model_name, pretrained=False, num_classes=num_classes)
                state_dict = torch.load(model_path, map_location="cpu")
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model_raw.load_state_dict(state_dict)
                model_raw = model_raw.to(device)

                temperature = tune_temperature(model_raw, val_loader, device)
                print(f"Learned temperature for {model_name} fold{fold}: T = {temperature:.4f}")

                # ✅ metrics 저장
                key = f"{model_name}_fold{fold}"
                if os.path.exists(args.metrics_file):
                    with open(args.metrics_file, "r") as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
                if "temperature" not in metrics:
                    metrics["temperature"] = {}
                metrics["temperature"][key] = round(temperature, 5)
                with open(args.metrics_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                print(f"✅ Temperature metrics updated in {args.metrics_file}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
