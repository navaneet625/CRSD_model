import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRSDMemoryFusion(nn.Module):
    def __init__(self, d_h, enable_similarity=False, enable_attention=False):
        super().__init__()
        self.d_h = d_h
        self.enable_similarity = enable_similarity
        self.enable_attention = enable_attention

        # learnable mix strength
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.mem_gate = nn.Linear(d_h, 1)

        if enable_similarity:
            self.sim_temp = nn.Parameter(torch.tensor(1.0))

        # optional cross-attention
        if enable_attention:
            self.q_proj = nn.Linear(d_h, d_h)
            self.k_proj = nn.Linear(d_h, d_h)
            self.v_proj = nn.Linear(d_h, d_h)
            self.attn_alpha = nn.Parameter(torch.tensor(0.25))

    def forward(self, y_ssm, h_mem_seq):
        """
        y_ssm: (B, T, d_h)
        h_mem_seq: (B, T, d_h)
        """
        eps = 1e-6  # numeric guard

        # --- A: normalized residual fusion ---
        y_norm = F.layer_norm(y_ssm, [self.d_h])
        h_norm = F.layer_norm(torch.tanh(h_mem_seq), [self.d_h])
        gate = torch.sigmoid(self.mem_gate(y_norm))
        # optional clamp to avoid runaway alpha during long training
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        y_out = y_norm + alpha * gate * h_norm

        # --- B: optional semantic similarity weighting ---
        if self.enable_similarity:
            denom = self.sim_temp * math.sqrt(self.d_h) + eps
            sim = (y_out * h_norm).sum(-1, keepdim=True) / denom
            w = torch.softmax(sim, dim=1)
            y_out = (1 - w) * y_out + w * h_norm

        # --- C: optional cross-attention for contextual recall ---
        if self.enable_attention:
            Q = self.q_proj(y_out)
            K = self.k_proj(h_norm)
            V = self.v_proj(h_norm)
            attn = torch.softmax(Q @ K.transpose(-2, -1) /
                                 (math.sqrt(self.d_h) + eps), dim=-1)
            y_out = y_out + self.attn_alpha * (attn @ V)

        return y_out