import torch
import torch.nn as nn

from .encoders import VisionEncoder, TextEncoder, AudioEncoder, AudioProjection
from .meter_fuser import METERFuser
from .hgf import HierarchicalGatedFusion
from .memory import MemoryAugmentedRefinement
from .heads import PredictionHeads

class CARMDD(nn.Module):
    """
    Full CARMDD model.
    forward(vision, audio, text) -> {"logits": [B], "severity": [B] or None}
    - vision: [B, T, 3, H, W]
    - audio:  [B, T, 1024, 128]  (AST spectrogram)
    - text:   (input_ids [B, T, L], attention_mask [B, T, L]) or None
    Config flags (meter, gating, memory, bge_m3) enable/disable modules for ablation.
    """
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.cfg = cfg
        self.use_text = cfg.get("use_text", True)
        self.use_regression = cfg.get("use_regression", False)

        self.vision = VisionEncoder(cfg["vit_name"])
        self.audio = AudioEncoder(cfg["ast_name"])
        self.audio_proj = AudioProjection(self.audio.dim, d)
        self.v_proj = nn.Linear(self.vision.dim, d)

        if self.use_text and cfg.get("bge_m3", True):
            self.text = TextEncoder(cfg["bge_m3_name"])
            self.t_proj = nn.Linear(self.text.dim, d)
        else:
            self.text = None
            self.t_proj = None

        self.meter = METERFuser(d) if (self.use_text and cfg.get("meter", True)) else None

        self.hgf = HierarchicalGatedFusion(
            d_model=d, n_heads=cfg.get("n_heads", 8),
            segment_len=cfg.get("segment_len", 64),
            segment_stride=cfg.get("segment_stride", 32),
            use_gating=cfg.get("gating", True),
        )
        self.memory = MemoryAugmentedRefinement(
            d_model=d, n_memory=cfg.get("n_memory", 64), top_k=cfg.get("top_k", 8),
        ) if cfg.get("memory", True) else None
        self.heads = PredictionHeads(d, use_regression=self.use_regression)

    def forward(self, vision, audio, text=None):
        F_v = self.v_proj(self.vision(vision))          # [B, T, D]
        F_a = self.audio_proj(self.audio(audio))        # [B, T, D]

        if self.text is not None and text is not None:
            ids, mask = text
            F_t = self.t_proj(self.text(ids, mask))     # [B, T, D]
        else:
            # AVEC 2019 path: no text; broadcast audio as pseudo-text for HGF shape compat
            F_t = F_a

        if self.meter is not None and self.text is not None and text is not None:
            F_v, F_t = self.meter(F_v, F_t)

        F_fuse, _, _, _ = self.hgf(F_v, F_a, F_t)
        F_refined = self.memory(F_fuse) if self.memory is not None else F_fuse
        logits, severity = self.heads(F_refined)
        return {"logits": logits, "severity": severity}