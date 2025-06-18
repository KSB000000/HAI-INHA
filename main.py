import argparse
import os
import json
import numpy as np
import torch
from datasets import CustomImageDataset
from torch.utils.data import  DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from timm import create_model
from transforms import get_train_transform
from transforms import get_val_transform
from collate import cutmix_collate_fn
from train import train_model
from calibration import tune_temperature
from seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=bool, default=True, help='Enable temperature scaling')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for CV')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--train_root', type=str, default="./data/train", help='Train root')
    parser.add_argument('--save_dir', type=str, default="./checkpoints", help='Directory of checkpoint')
    parser.add_argument('--metrics_file', type=str, default="val_metrics1.json", help='Metric file name')
    return parser.parse_args()


def main(args):
    SEED = 42
    seed_everything(SEED) # Seed 고정

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = args.train_root


    full_dataset = CustomImageDataset(train_root, transform=False)
    print(f"총 이미지 수: {len(full_dataset)}")

    targets = [label for _, label in full_dataset.samples]
    class_names = full_dataset.classes
    num_classes = len(class_names)


    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=SEED)


    model_names = [
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "regnety_1280.swag_ft_in1k"
    ]

    # fold loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n=== Fold {fold+1}/{args.num_folds} ===")

        for model_name in model_names:
            print(f"▶ Training model: {model_name} (Fold {fold})")

            # transform은 fold마다 새로 적용
            train_dataset = Subset(full_dataset, train_idx)
            val_dataset = Subset(full_dataset, val_idx)
            if model_name == "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k":
                train_dataset.dataset.transform = get_train_transform(448)
                val_dataset.dataset.transform = get_val_transform(448)
                if fold == 0:
                    lr = 5e-6
                else:
                    lr = 1e-5
            else:
                train_dataset.dataset.transform = get_train_transform(384)
                val_dataset.dataset.transform = get_val_transform(384)
                lr = 1e-5

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, collate_fn=cutmix_collate_fn, pin_memory=True,
                                    persistent_workers=True, prefetch_factor=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True,
                                    persistent_workers=True, prefetch_factor=4)

            # 모델 로드
            model = create_model(model_name, pretrained=True, num_classes=num_classes).to(device)


            # 학습 loop
            train_model(model, train_loader, val_loader, model_name, fold,
                        num_classes, device, epochs=args.epochs, lr=lr,
                        save_dir=args.save_dir, metrics_file=args.metrics_file)

            # Calibration
            if args.calibration:
                print(f"Calibrating model: {model_name} (Fold {fold})")
                
                # 모델 불러오기 (학습된 상태)
                model_path = os.path.join(args.save_dir, f"{model_name}_fold{fold}.pth")
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                model.eval()  # 꼭 eval 모드로!

                # Temperature tuning
                temperature = tune_temperature(model, val_loader, device)
                print(f"Learned temperature for {model_name} fold{fold}: T = {temperature:.4f}")

                # 📌 metrics_file 업데이트
                key = f"{model_name}_fold{fold}"
                
                # 기존 metrics 파일 불러오기 (logloss, accuracy 유지)
                if os.path.exists(args.metrics_file):
                    with open(args.metrics_file, "r") as f:
                        metrics = json.load(f)
                else:
                    metrics = {"logloss": {}, "accuracy": {}, "temperature": {}}  # 처음 생성 시 전체 key 포함

                # temperature 값 업데이트
                if "temperature" not in metrics:
                    metrics["temperature"] = {}
                metrics["temperature"][key] = round(temperature, 5)

                # 업데이트된 metrics 저장
                with open(args.metrics_file, "w") as f:
                    json.dump(metrics, f, indent=4)

                print(f"✅ Temperature metrics updated in {args.metrics_file}")


if __name__=='__main__':
    args = parse_args()
    main(args)
