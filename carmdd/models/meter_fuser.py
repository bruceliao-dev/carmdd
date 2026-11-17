import torch
import torch.nn as nn
import torch.nn.functional as F

class METERFuser(nn.Module):
    """
    Simplified METER-style dual-stream cross-attention fusion.
    Eq. align: F_v^align, F_t^align = METER-Fuser(F_v, F_t)
    If use_text is False, this module is bypassed at the CARMDD level.
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.v_proj = nn.Linear(d_model, d_model)
        self.t_proj = nn.Linear(d_model, d_model)
        self.v2t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.t2v = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, F_v, F_t):
        v = self.v_proj(F_v)
        t = self.t_proj(F_t)
        v_att, _ = self.v2t(query=v, key=t, value=t)
        t_att, _ = self.t2v(query=t, key=v, value=v)
        v = self.norm_v(v + v_att)
        t = self.norm_t(t + t_att)
        v = self.norm_v(v + self.ffn_v(v))
        t = self.norm_t(t + self.ffn_t(t))
        return v, t