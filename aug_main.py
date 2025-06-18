import argparse
import os
import json
import numpy as np
import torch
from aug_datasets import CustomImageDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from timm import create_model
from aug_transforms import get_train_transform, get_val_transform
from collate import cutmix_collate_fn
from train import train_model
from calibration import tune_temperature
from seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=bool, default=True)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--train_root', type=str, default="./data/train")
    parser.add_argument('--save_dir', type=str, default="./checkpoint_all")
    parser.add_argument('--metrics_file', type=str, default="val_all.json")
    return parser.parse_args()


def main(args):
    SEED = 42
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset_no_crop = CustomImageDataset(args.train_root, transform=None, is_test=False)
    targets = [sample[1] for sample in full_dataset_no_crop.samples]
    class_names = full_dataset_no_crop.classes
    num_classes = len(class_names)
    print(f"총 원본 이미지 수: {len(full_dataset_no_crop)}")

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=SEED)

    # ✅ 각 모델에 맞는 input_size 정의
    model_configs = {
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k": 448,
        "regnety_1280.swag_ft_in1k": 384
    }


    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n=== Fold {fold+1}/{args.num_folds} ===")

        for model_name, input_size in model_configs.items():
            print(f"\u25b6 Training model: {model_name} (Fold {fold})")

            # ✅ Transform 생성 시 input_size 사용
            val_samples = [(*full_dataset_no_crop.samples[i], 'original') for i in val_idx]
            val_dataset = CustomImageDataset(args.train_root, transform=get_val_transform(input_size), is_test=False)
            val_dataset.samples = val_samples

            train_samples = [full_dataset_no_crop.samples[i] for i in train_idx]
            train_samples_aug = []
            for img_path, label in train_samples:
                for mode in ['original', 'crop', 'rotate']:
                    train_samples_aug.append((img_path, label, mode))

            train_dataset = CustomImageDataset(args.train_root, transform=get_train_transform(input_size), is_test=False)
            train_dataset.samples = train_samples_aug

            print(f"Train 샘플 수: {len(train_dataset)}")
            print(f"Val 샘플 수: {len(val_dataset)}")

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

            model = create_model(model_name, pretrained=True, num_classes=num_classes)
            # if torch.cuda.device_count() > 1:
            #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            #     model = torch.nn.DataParallel(model)
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
                print(f"\u2705 Temperature metrics updated in {args.metrics_file}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
