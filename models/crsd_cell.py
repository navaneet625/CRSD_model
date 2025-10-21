import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .crsd_kcm import KCMemory
from .fusion import CRSDMemoryFusion


def apply_rope_shared(x, sin, cos):
    """Apply rotary embeddings (shared RoPE)"""
    if sin is None or cos is None:
        return x
    B, T, D = x.shape
    H = sin.shape[-1]
    half = min(H, D // 2)
    if half == 0:
        return x
    current_T = min(T, sin.shape[1])
    sin, cos = sin[:, :current_T, :half], cos[:, :current_T, :half]
    x_input = x[:, :current_T, :]
    x1, x2 = x_input[..., :half], x_input[..., half:2 * half]
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if D > 2 * half:
        rot = torch.cat([rot, x_input[..., 2 * half:]], dim=-1)
    if T > current_T:
        rot = torch.cat([rot, x[:, current_T:, :]], dim=1)
    return rot


class CRSDCell(nn.Module):
    """Core CRSD state-space + key-value memory cell"""

    def __init__(self,
                 d_x: int,
                 d_h: int,
                 N: int,
                 d_k: int,
                 d_v: int,
                 rank_kcm: int = 64,
                 kcm_consolidation_rate: float = 0.10,
                 kcm_reduce: str = "sum",
                 use_rope: bool = True,
                 auto_write: bool = True,
                 res_dropout_p: float = 0.1,
                 memory_dtype=torch.float32,
                 shared_mem: KCMemory | None = None,
                 debug: bool = False,
                 stateful: bool = True):
        super().__init__()
        self.d_x, self.d_h, self.N = int(d_x), int(d_h), int(N)
        self.use_rope, self.stateful = use_rope, stateful
        self.d_k, self.d_v = d_k, d_v

        # SSM parameters
        self.A_log = nn.Parameter(torch.ones(self.d_h, self.N) * -5.0)
        self.delta_proj = nn.Linear(d_x, 1)
        self.B_gen = nn.Linear(d_x, d_h * self.N)
        self.C_gen = nn.Linear(d_x, d_h * self.N)

        # Residual/output path
        self.D_proj = nn.Linear(d_x, d_h)
        self.D_scale = nn.Parameter(torch.ones(d_h))

        # Memory interface
        self.key_in = nn.Linear(d_h, d_k)
        self.val_in = nn.Linear(d_h, d_v)
        self.recall_map = nn.Linear(d_v, d_h)
        self.mem_gate = nn.Linear(d_h, 1)

        # Normalization + dropout
        self.h_norm = nn.LayerNorm(d_h)
        self.dropout = nn.Dropout(res_dropout_p) if res_dropout_p > 0 else nn.Identity()

        # Shared memory
        self._kcm: KCMemory = shared_mem or KCMemory(
            d_k=d_k, d_v=d_v, rank=rank_kcm, memory_dtype=memory_dtype,
            kcm_consolidation_rate=kcm_consolidation_rate, kcm_reduce=kcm_reduce,
        )

        self.fusion_module = CRSDMemoryFusion(d_h=self.d_h, enable_similarity=False, enable_attention=False)
        self._last_h = None

    def reset_state(self):
        self._last_h = None

    def forward(self, x_seq, h0=None, train_mode=True, rope_cache=None):
        B, T, _ = x_seq.shape
        device = x_seq.device

        # Move KCM to correct device/dtype
        if self._kcm.M.device != device or self._kcm.M.dtype != self._kcm.dtype:
            with torch.no_grad():
                self._kcm.M = self._kcm.M.to(device=device, dtype=self._kcm.dtype)

        # SSM input normalization + optional RoPE
        x_ssm = self.h_norm(x_seq)
        if self.use_rope and rope_cache is not None:
            x_ssm = apply_rope_shared(x_ssm, *rope_cache)

        Bsz, Tlen, _ = x_ssm.shape
        delta = F.softplus(self.delta_proj(x_ssm))          # (B,T,1)
        delta = torch.clamp(delta, max=0.5)
        A_cont = -torch.exp(self.A_log).to(x_seq.dtype).unsqueeze(0).unsqueeze(0)
        
        DeltaA = delta.unsqueeze(-1) * A_cont               
        A_bar  = torch.exp(DeltaA)                          
        eps = 1e-4
        injection = torch.where(
            torch.abs(A_cont) > eps,
            torch.expm1(DeltaA) / A_cont, 
            delta.unsqueeze(-1) 
        )

        B_cont = F.silu(self.B_gen(x_ssm)).view(Bsz, Tlen, self.d_h, self.N)
        B_cont = B_cont - B_cont.mean(dim=(0,1), keepdim=True)
        B_cont = 0.5 * B_cont
        B_bar = injection * B_cont                          

        A_bar = torch.clamp(A_bar, 0.0, 1.0)
        B_bar = torch.nan_to_num(B_bar, nan=0.0, posinf=1e3, neginf=-1e3)
        h_seq = self._recurrent_fallback(A_bar, B_bar, x_ssm, h0, self.stateful)

        C_coeff = F.silu(self.C_gen(x_ssm)).view(Bsz, Tlen, self.d_h, self.N)
        h_out = (h_seq * C_coeff).sum(dim=-1)

        y_ssm = h_out + self.D_scale * self.D_proj(x_seq)

        # Memory read/fuse
        k, v = self.key_in(y_ssm), self.val_in(y_ssm)
        if self.use_rope and rope_cache is not None:
            k = apply_rope_shared(k, *rope_cache)
        v_hat_seq = self._kcm.batch_recall(k)
        h_mem_seq = self.recall_map(v_hat_seq)
        y_ssm = self.fusion_module(y_ssm, h_mem_seq)

        # Cache last hidden state
        if self.stateful:
            self._last_h = h_seq[:, -1].detach()

        if train_mode:
            return self.dropout(y_ssm), h_seq, k, v
        return self.dropout(y_ssm), h_seq

    def _recurrent_fallback(self, A_bar, B_bar,x_ssm, h0, stateful):
        B, T, D, N = A_bar.shape
        dtype = A_bar.dtype
        device = A_bar.device
    
        if stateful and self._last_h is not None:
            h_prev = self._last_h
        elif h0 is not None:
            h_prev = h0
        else:
            h_prev = torch.zeros(B, D, N, dtype=dtype, device=device)
 
        A_shift = torch.cat([torch.ones(B, 1, D, N, device=device, dtype=dtype), A_bar[:, :-1]], dim=1)
        logA_shift = torch.log(torch.clamp(A_shift, min=1e-6))
        log_prefix_prod_shift = torch.cumsum(logA_shift, dim=1)
        prefix_prod_shift = torch.exp(log_prefix_prod_shift)

        weighted_B = B_bar * prefix_prod_shift
        cumsum_weighted = torch.cumsum(weighted_B, dim=1)
        h_seq = cumsum_weighted / prefix_prod_shift 
        h_seq += prefix_prod_shift * h_prev.unsqueeze(1)
        return h_seq
