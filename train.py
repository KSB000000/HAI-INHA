import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
import json
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import log_loss
from loss import combined_criterion


# 📝 전체 모델별 metric 기록 (global dict)
val_metrics_logloss = {}
val_metrics_accuracy = {}

def train_model(model, train_loader, val_loader, model_name, fold, num_classes, device,
                epochs=30, lr=1e-4, save_dir="./checkpoints", early_stop_patience=5,
                metrics_file="val_metrics.json"):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_logloss = float('inf')
    best_acc = 0.0
    no_improve_epochs = 0  # early stopping counter
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            images = images.to(device)
            if len(targets) == 3:
                y1, y2, lam = targets
                targets = (y1.to(device), y2.to(device), lam)
            else:
                targets = targets.to(device)

            optimizer.zero_grad()
                
            with autocast():
                outputs = model(images)
                loss = combined_criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()


        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs, all_labels = [], []


        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Val] Epoch {epoch}"):
                images = images.to(device)
                labels = labels.to(device)

                with autocast():
                    outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(probs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_probs).numpy()
        val_logloss = log_loss(y_true, y_pred, labels=list(range(num_classes)))
        val_acc = correct / total

        print(f"Epoch {epoch}: "
            f"Train Loss = {train_loss/len(train_loader):.4f}, "
            f"Val Loss = {val_loss/len(val_loader):.4f}, "
            f"LogLoss = {val_logloss:.5f}, "
            f"Accuracy = {val_acc*100:.2f}%")

        # Check for improvement
        if val_logloss < best_logloss:
            best_logloss = val_logloss
            best_acc = val_acc
            no_improve_epochs = 0
            model_path = os.path.join(save_dir, f"{model_name}_fold{fold}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model: {model_path} (logloss={val_logloss:.5f})")
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement for {no_improve_epochs} epoch(s).")

        # Early stopping trigger
        if no_improve_epochs >= early_stop_patience:
            print(f"🛑 Early stopping triggered after {epoch} epochs (patience={early_stop_patience})")
            break

    # 🎯 Fold 종료 후 metric 기록
    key = f"{model_name}_fold{fold}"
    val_metrics_logloss[key] = round(best_logloss, 5)
    val_metrics_accuracy[key] = round(best_acc, 5)

    # 🎯 매 fold마다 metric 저장
    metrics = {"logloss": val_metrics_logloss, "accuracy": val_metrics_accuracy}
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Validation metrics updated in {metrics_file}")
