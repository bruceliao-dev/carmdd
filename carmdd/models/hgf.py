import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLevelGatedAttention(nn.Module):
    """Eq. gate + Eq. bidirectional, feature level."""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(3 * d_model, 3)
        self.cross_va = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_vt = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_av = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_at = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_tv = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_ta = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(3 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, F_v, F_a, F_t):
        # all: [B, T, D]
        g = F.softmax(self.gate(torch.cat([F_v, F_a, F_t], dim=-1)), dim=-1)  # [B, T, 3]
        F_v = F_v * g[..., 0:1]
        F_a = F_a * g[..., 1:2]
        F_t = F_t * g[..., 2:3]

        va, _ = self.cross_va(F_v, F_a, F_a)
        vt, _ = self.cross_vt(F_v, F_t, F_t)
        av, _ = self.cross_av(F_a, F_v, F_v)
        at, _ = self.cross_at(F_a, F_t, F_t)
        tv, _ = self.cross_tv(F_t, F_v, F_v)
        ta, _ = self.cross_ta(F_t, F_a, F_a)

        # Paper Eq. bidirectional sums over modalities as query
        F_fuse = torch.cat([va + vt, av + at, tv + ta], dim=-1)
        F_fuse = self.norm(self.out(F_fuse))
        return F_fuse  # [B, T, D]

class GlobalAttentionPooling(nn.Module):
    """Eq. global_attn."""
    def __init__(self, d_model):
        super().__init__()
        self.V = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: [B, T, D]
        scores = self.w(torch.tanh(self.V(x))).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", alpha, x), alpha

class HierarchicalGatedFusion(nn.Module):
    """
    Full HGF: feature-level, segment-level, global-level; then Eq. fuse_concat.
    """
    def __init__(self, d_model, n_heads=8, segment_len=64, segment_stride=32, use_gating=True):
        super().__init__()
        self.feature_level = FeatureLevelGatedAttention(d_model, n_heads)
        self.segment_level = FeatureLevelGatedAttention(d_model, n_heads)
        self.global_pool = GlobalAttentionPooling(d_model)
        self.segment_len = segment_len
        self.segment_stride = segment_stride
        self.use_gating = use_gating
        # project 3D -> D after concat (Eq. fuse_concat)
        self.concat_proj = nn.Linear(3 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _segments(self, x):
        # x: [B, T, D] -> [B, S, L, D]
        B, T, D = x.shape
        L, S = self.segment_len, self.segment_stride
        if T < L:
            pad = L - T
            x = F.pad(x, (0, 0, 0, pad))
            T = L
        segs = x.unfold(1, L, S)  # [B, S, D, L]
        return segs.permute(0, 1, 3, 2)  # [B, S, L, D]

    def forward(self, F_v, F_a, F_t, mask=None):
        # Feature level
        F_feat = self.feature_level(F_v, F_a, F_t)  # [B, T, D]

        # Segment level: average-pool within segment, then same gated attention
        sv, sa, st = self._segments(F_v), self._segments(F_a), self._segments(F_t)
        sv, sa, st = sv.mean(2), sa.mean(2), st.mean(2)  # [B, S, D]
        F_seg = self.segment_level(sv, sa, st)  # [B, S, D]

        # Upsample segment -> time
        B, T, D = F_feat.shape
        S = F_seg.shape[1]
        idx = torch.arange(T, device=F_feat.device) // self.segment_stride
        idx = idx.clamp(max=S - 1)
        F_seg_up = F_seg[:, idx]  # [B, T, D]

        # Global
        F_global, _ = self.global_pool(F_feat, mask)  # [B, D]
        F_global_b = F_global.unsqueeze(1).expand(-1, T, -1)

        # Eq. fuse_concat
        F_cat = torch.cat([F_feat, F_seg_up, F_global_b], dim=-1)  # [B, T, 3D]
        F_fuse = self.norm(self.concat_proj(F_cat))  # [B, T, D]
        return F_fuse, F_feat, F_seg, F_global