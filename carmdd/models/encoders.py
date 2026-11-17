import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel

class VisionEncoder(nn.Module):
    """ViT backbone; output F_v in R^{T x D_v}."""
    def __init__(self, name="google/vit-base-patch16-224", freeze=False):
        super().__init__()
        self.vit = ViTModel.from_pretrained(name)
        self.dim = self.vit.config.hidden_size
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        # pixel_values: [B, T, 3, H, W]
        B, T = pixel_values.shape[:2]
        x = pixel_values.view(B * T, *pixel_values.shape[2:])
        out = self.vit(pixel_values=x).last_hidden_state.mean(dim=1)  # [B*T, D]
        return out.view(B, T, -1)

class TextEncoder(nn.Module):
    """BGE-M3 (or any HF encoder); output F_t in R^{T x D_t}.
    Input is a list of token-id sequences per sample (padded to T)."""
    def __init__(self, name="BAAI/bge-m3", freeze=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.dim = self.model.config.hidden_size
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # input_ids: [B, T, L]
        B, T, L = input_ids.shape
        ids = input_ids.view(B * T, L)
        mask = attention_mask.view(B * T, L)
        out = self.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]  # CLS
        return out.view(B, T, -1)

class AudioEncoder(nn.Module):
    """HTS-AT; output F_a in R^{T x D_a}."""
    def __init__(self, name="MIT/ast-finetuned-audioset-10-10-0.4593", freeze=False):
        super().__init__()
        from transformers import ASTModel
        self.ast = ASTModel.from_pretrained(name)
        self.dim = self.ast.config.hidden_size
        if freeze:
            for p in self.ast.parameters():
                p.requires_grad = False

    def forward(self, spectrogram):
        # spectrogram: [B, T, 1024, 128]
        B, T = spectrogram.shape[:2]
        x = spectrogram.view(B * T, *spectrogram.shape[2:])
        out = self.ast(x).last_hidden_state.mean(dim=1)
        return out.view(B, T, -1)

class AudioProjection(nn.Module):
    """LayerNorm(F_a W_a) — Eq. audio_align."""
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.proj(x))