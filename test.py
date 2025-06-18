import argparse
import torch
import numpy as np
import pandas as pd
import json
from datasets import CustomImageDataset
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from transforms import get_test_transform, get_val_transform
from seed import seed_everything
from timm import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=bool, default=False, help='Apply temperature scaling')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for test loader')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--metric_files', nargs=2, default=['val_all.json', 'val.json'], help='List of metric jsons')
    parser.add_argument('--checkpoint_dirs', nargs=2, default=['checkpoint_all', 'checkpoints'], help='List of checkpoint dirs')
    parser.add_argument('--output_csv', type=str, default='combined_submission.csv', help='Submission file')
    parser.add_argument('--train_root', type=str, default='./data/train', help='Train root')
    parser.add_argument('--test_root', type=str, default='./data/test', help='Test root')
    parser.add_argument('--top_k', type=int, default=2, help="Number of RegNet's top folds")
    return parser.parse_args()


def merge_metric_jsons(val_eva_path='val_eva.json', val_reg_path='val_reg.json', merged_path='val.json'):
    def load_json(path):
        if not os.path.exists(path):
            print(f"⚠️  {path} not found. Skipping.")
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    eva = load_json(val_eva_path)
    reg = load_json(val_reg_path)

    merged = {'logloss': {}, 'temperature': {}}
    for k in ['logloss', 'temperature']:
        merged[k].update(eva.get(k, {}))
        merged[k].update(reg.get(k, {}))

    with open(merged_path, 'w') as f:
        json.dump(merged, f, indent=4)
    print(f"✅ Merged metric json saved to '{merged_path}'")


def main(args):
    SEED = 42
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    merge_metric_jsons()

    # class name list
    full_dataset = CustomImageDataset(args.train_root, transform=False)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    folds = 5

    model_names = [
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "regnety_1280.swag_ft_in1k"
    ]

    # Load metrics and temperature info
    all_entries = []
    for metric_file, ckpt_dir in zip(args.metric_files, args.checkpoint_dirs):
        with open(metric_file, 'r') as f:
            metrics = json.load(f)
        loglosses = metrics.get('logloss', {})
        temps = metrics.get('temperature', {})

        for model_name in model_names:
            fold_metrics = []
            for fold in range(folds):
                key = f"{model_name}_fold{fold}"
                if key in loglosses:
                    logloss = loglosses[key]
                    fold_metrics.append((key, model_name, fold, logloss, ckpt_dir, temps.get(key, 1.0)))

            fold_metrics.sort(key=lambda x: x[3])
            if model_name.startswith("eva"):
                all_entries.extend(fold_metrics)  # eva는 전체 사용
            else:
                all_entries.extend(fold_metrics[:args.top_k])  # regnet은 top_k만 사용

    print("\n🎯 Selected folds:")
    for key, model_name, fold, logloss, ckpt_dir, T in all_entries:
        print(f"{key} ({ckpt_dir}): logloss={logloss:.5f}, T={T:.3f}")

    # Compute weights
    logloss_values = np.array([x[3] for x in all_entries])
    inverse_logloss = 1 / logloss_values
    weights = inverse_logloss / inverse_logloss.sum()

    print("\n🎯 Soft voting weights:")
    for (key, model_name, fold, logloss, ckpt_dir, _), weight in zip(all_entries, weights):
        print(f"{key} ({ckpt_dir}): weight={weight:.4f}")

    # Inference
    test_preds = []
    for (key, model_name, fold, logloss, ckpt_dir, T), weight in zip(all_entries, weights):
        img_size = 448 if model_name.startswith("eva") else 384

        # ✅ transform 선택 기준
        if ckpt_dir == "checkpoint_all":
            transform_fn = get_test_transform(img_size)
        else:
            transform_fn = get_val_transform(img_size)

        test_dataset = CustomImageDataset(args.test_root, transform=transform_fn, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        model = create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(f"{ckpt_dir}/{key}.pth", map_location=device))
        model.to(device)
        model.eval()

        T = T if args.calibration else 1.0
        print(f"\n🚀 Inference started for {key} from {ckpt_dir} (T={T:.4f})")

        all_probs = []
        with torch.no_grad():
            for images in tqdm(test_loader, desc=key, leave=False):
                images = images.to(device)
                with autocast(device.type):
                    logits = model(images)
                logits = logits / T
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        test_preds.append(probs)

    # Final prediction
    test_preds = np.stack(test_preds, axis=0)  # (num_models, N, C)
    final_probs = np.average(test_preds, axis=0, weights=weights)

    # Save submission
    pred = pd.DataFrame(final_probs, columns=class_names)
    submission = pd.read_csv('../open/sample_submission.csv', encoding='utf-8-sig')
    class_columns = submission.columns[1:]
    pred = pred[class_columns]
    submission[class_columns] = pred.values
    submission.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ Submission file saved as '{args.output_csv}'")


if __name__ == '__main__':
    args = parse_args()
    main(args)
