import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes, alpha, adversarial=False):
        super().__init__()

        # alpha 값이 0에서 1 사이인지 검증
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f'Smoothing value should be between 0 and 1. Got: {alpha}')

        self.num_classes = num_classes
        self.alpha = alpha
        self.confidence = 1.0 - (alpha / (num_classes - adversarial))
        self.adversarial = 1 if adversarial else 0 # 0: standard label smoothing, 1: adversarial label smoothing 

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.alpha / (self.num_classes - self.adversarial))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))