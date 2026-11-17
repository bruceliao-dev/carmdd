import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    """
    L = alpha * BCE + beta * SmoothL1
    Paper Eq. total_loss. For LMVD, set alpha=1, beta=0.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, severity, labels, phq8):
        # logits: [B], severity: [B], labels: [B] in {0,1}, phq8: [B] or None
        loss = 0.0
        if self.alpha > 0:
            loss = loss + self.alpha * F.binary_cross_entropy_with_logits(logits, labels.float())
        if self.beta > 0 and severity is not None and phq8 is not None:
            loss = loss + self.beta * F.smooth_l1_loss(severity, phq8.float(), beta=1.0)
        return loss