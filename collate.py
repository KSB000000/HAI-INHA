import torch
import numpy as np


def cutmix_collate_fn(batch, alpha=1.0):
    """
    batch: list of (image, label)
    alpha: Beta 분포의 파라미터 (작을수록 혼합 비율 극단적)
    """
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # CutMix 적용 여부 (optional: 확률적으로 적용)
    apply_cutmix = True
    if not apply_cutmix:
        return images, labels

    # 샘플 쌍 생성
    indices = torch.randperm(images.size(0))
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    # lambda 샘플링 (beta 분포)
    lam = np.random.beta(alpha, alpha)
    
    # bounding box 위치 계산
    W = images.size(2)
    H = images.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 랜덤 center 위치
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 이미지 교환
    images[:, :, bby1:bby2, bbx1:bbx2] = shuffled_images[:, :, bby1:bby2, bbx1:bbx2]

    # 라벨 혼합
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    mixed_labels = (labels, shuffled_labels, lam)

    return images, mixed_labels
