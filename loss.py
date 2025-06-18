import torch.nn.functional as F


def combined_criterion(pred, targets):
    y1, y2, lam = targets
    return lam * F.cross_entropy(pred, y1) + (1 - lam) * F.cross_entropy(pred, y2)
