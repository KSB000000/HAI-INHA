import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize


class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature


def tune_temperature(model, val_loader, device):
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    def loss_fn(temp):
        temp = torch.tensor(temp, requires_grad=False)
        scaled_logits = logits / temp
        return F.cross_entropy(scaled_logits, labels).item()

    res = minimize(loss_fn, x0=[1.5], bounds=[(0.5, 5.0)], method='L-BFGS-B')
    best_temp = res.x[0]

    return best_temp
