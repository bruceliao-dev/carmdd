import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScalePooling(nn.Module):
    """Windows of 32, 64, full-sequence; concat pooled features."""
    def __init__(self, d_model, windows=(32, 64, None)):
        super().__init__()
        self.windows = windows
        self.proj = nn.Linear(d_model * len(windows), d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        pooled = []
        for w in self.windows:
            if w is None or w >= T:
                pooled.append(x.mean(1))
            else:
                n = T // w
                if n == 0:
                    pooled.append(x.mean(1))
                else:
                    trimmed = x[:, :n * w].view(B, n, w, D).mean(2).mean(1)
                    pooled.append(trimmed)
        return self.proj(torch.cat(pooled, dim=-1))  # [B, D]

class PredictionHeads(nn.Module):
    def __init__(self, d_model, use_regression=True, dropout=0.1):
        super().__init__()
        self.pool = MultiScalePooling(d_model)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.use_regression = use_regression
        if use_regression:
            self.reg = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, x):
        z = self.pool(x)
        logits = self.cls(z).squeeze(-1)
        severity = self.reg(z).squeeze(-1) if self.use_regression else None
        return logits, severity