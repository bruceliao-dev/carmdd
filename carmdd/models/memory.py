import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryAugmentedRefinement(nn.Module):
    """
    External memory M in R^{N_m x D}; read (top-k), reason (Transformer), write (erase/add).
    Eq. read_scores, read_retrieve, reason_update, write_vectors, memory_update.
    """
    def __init__(self, d_model, n_memory=64, top_k=8, n_heads=4, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_memory = n_memory
        self.top_k = top_k
        self.memory = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)

        self.reason = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        # project concat back to d_model
        self.reason_in = nn.Linear(2 * d_model, d_model)

        self.erase = nn.Linear(d_model, d_model)
        self.add = nn.Linear(d_model, d_model)

    def forward(self, F_fuse):
        # F_fuse: [B, T, D]
        B, T, D = F_fuse.shape
        M = self.memory  # [N, D]
        outputs = []

        for t in range(T):
            x = F_fuse[:, t]  # [B, D]
            # Read: Eq. read_scores
            scores = F.softmax(x @ M.t(), dim=-1)  # [B, N]
            # top-k: Eq. read_retrieve
            k = min(self.top_k, self.n_memory)
            topv, topi = scores.topk(k, dim=-1)  # [B, k]
            topv = topv / (topv.sum(-1, keepdim=True) + 1e-8)
            # gather memory rows
            M_exp = M.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
            idx = topi.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
            retrieved = torch.gather(M_exp, 1, idx)  # [B, k, D]
            r = torch.einsum("bk,bkd->bd", topv, retrieved)  # [B, D]

            # Reason: Eq. reason_update
            combined = self.reason_in(torch.cat([x, r], dim=-1)).unsqueeze(1)  # [B, 1, D]
            refined = self.reason(combined).squeeze(1)  # [B, D]
            outputs.append(refined)

            # Write: Eq. write_vectors + memory_update
            e = torch.sigmoid(self.erase(refined))  # [B, D]
            a = torch.tanh(self.add(refined))       # [B, D]
            # aggregate over batch (paper uses per-sample write; here we use mean)
            e_mean = e.mean(0, keepdim=True)  # [1, D]
            a_mean = a.mean(0, keepdim=True)  # [1, D]
            ones = torch.ones(self.n_memory, 1, device=M.device)  # [N, 1]
            self.memory.data = self.memory.data * (1 - ones @ e_mean) + ones @ a_mean

        return torch.stack(outputs, dim=1)  # [B, T, D]