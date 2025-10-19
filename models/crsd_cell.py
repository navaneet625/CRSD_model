import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .crsd_kcm import KCMemory

def apply_rope_shared(x, sin, cos):
    """
    Apply cached RoPE terms (B, T, D) to input (B, T, D).
    """
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
        self.d_x, self.d_h = int(d_x), int(d_h)
        self.N = int(N)
        self.use_rope = bool(use_rope)
        self.auto_write = bool(auto_write)
        self.debug = bool(debug)
        self.stateful = bool(stateful)
        self.d_k, self.d_v = d_k, d_v

        self.A_log = nn.Parameter(torch.ones(self.d_h, self.N) * -5.0)

        # Input-dependent parameter generators → per-(D,N) tensors
        self.delta_proj = nn.Linear(d_x, 1)              # (B,T,1)
        self.B_gen      = nn.Linear(d_x, d_h * self.N)   # (B,T,D*N) → view to (B,T,D,N)
        self.C_gen      = nn.Linear(d_x, d_h * self.N)   # (B,T,D*N) → view to (B,T,D,N)

        # Output / residual path
        self.D_proj = nn.Linear(d_x, d_h)
        self.D_scale = nn.Parameter(torch.ones(d_h))

        # --- Memory Interface (KCM) ---
        self.key_in = nn.Linear(d_h, d_k)
        self.val_in = nn.Linear(d_h, d_v)
        self.recall_map = nn.Linear(d_v, d_h)
        self.mem_gate = nn.Linear(d_h, 1)

        # Norm + Dropout
        self.h_norm = nn.LayerNorm(d_h)
        self.dropout = nn.Dropout(res_dropout_p) if res_dropout_p > 0 else nn.Identity()
        
        # KCMemory (shared or local)
        self._kcm: KCMemory = shared_mem or KCMemory(
            d_k=d_k, d_v=d_v, rank=rank_kcm, memory_dtype=memory_dtype,
            debug=debug, kcm_consolidation_rate=kcm_consolidation_rate,
            kcm_soft_decay=0.0, kcm_reduce=kcm_reduce,
        )

        # Cached recurrent state (for stateful inference)
        self._last_h = None
        self._pending_write = None 
        
    def reset_state(self):
        self._last_h = None

    def forward(self, x_seq, h0=None, train_mode=True, rope_cache=None):
        B, T, D_x = x_seq.shape
        device = x_seq.device
        # --- Ensure memory on correct device & dtype ---
        if self._kcm.M.device != x_seq.device or self._kcm.M.dtype != self._kcm.dtype:
            self._kcm.M = self._kcm.M.to(device=x_seq.device, dtype=self._kcm.dtype)
        
                
        if self._kcm.M.device != device:
            self._kcm.M = self._kcm.M.to(device)

        x_ssm = self.h_norm(x_seq)
        if self.use_rope and rope_cache is not None:
            x_ssm = apply_rope_shared(x_ssm, *rope_cache)
        
        Bsz, Tlen, _ = x_ssm.shape
        delta = F.softplus(self.delta_proj(x_ssm))          # (B,T,1)
        delta = torch.clamp(delta, max=0.5)
        A_cont = -torch.exp(self.A_log).to(x_seq.dtype).unsqueeze(0).unsqueeze(0)
        
        DeltaA = delta.unsqueeze(-1) * A_cont               # (B,T,D,N)
        A_bar  = torch.exp(DeltaA)                          # e^{ΔA}
        
        eps = 1e-4
        injection = torch.where(
            torch.abs(A_cont) > eps,
            torch.expm1(DeltaA) / A_cont, 
            delta.unsqueeze(-1) 
        )

        B_cont = F.silu(self.B_gen(x_ssm)).view(Bsz, Tlen, self.d_h, self.N)
        B_cont = B_cont - B_cont.mean(dim=(0,1), keepdim=True)
        B_cont = 0.5 * B_cont
        B_bar = injection * B_cont                          # (B,T,D,N)

        A_bar = torch.clamp(A_bar, 0.0, 1.0)
        B_bar = torch.nan_to_num(B_bar, nan=0.0, posinf=1e3, neginf=-1e3)
        h_seq = self._recurrent_fallback(A_bar, B_bar, x_ssm, h0, self.stateful)  # (B,T,D,N)

        C_coeff = F.silu(self.C_gen(x_ssm)).view(Bsz, Tlen, self.d_h, self.N)  # (B,T,D,N)
        h_out = (h_seq * C_coeff).sum(dim=-1)  # (B,T,D)

        # Residual path
        y_ssm = h_out + self.D_scale * self.D_proj(x_seq)
        
        # 5) KCM Integration (parallel over time; batch-safe)
        k = self.key_in(y_ssm)  # (B, T, d_k)
        v = self.val_in(y_ssm)  # (B, T, d_v)
        
        if self.use_rope and rope_cache is not None:
            # apply_rope_shared expects (B, T, D)
            k = apply_rope_shared(k, *rope_cache)  # stays (B, T, d_k)
        
        # Parallel recall per sample (vectorized over T, tiny loop over B)
        Bsz = k.shape[0]
        v_hat_seq_list = []
        for b in range(Bsz):
            v_hat_b = self._kcm.parallel_recall(k[b])   # (T, d_v)
            v_hat_seq_list.append(v_hat_b.unsqueeze(0)) # (1, T, d_v)
        v_hat_seq = torch.cat(v_hat_seq_list, dim=0)    # (B, T, d_v)
        
        h_mem_seq = self.recall_map(v_hat_seq)          # (B, T, d_h)
        mem_gate  = torch.sigmoid(self.mem_gate(y_ssm)) # (B, T, 1)
        y_ssm     = y_ssm + mem_gate * h_mem_seq        # (B, T, d_h)
        
        # Parallel write per sample (vectorized over T)
        if train_mode and self.auto_write:
            with torch.no_grad():
                for b in range(Bsz):
                    self._kcm.parallel_update(k[b].detach(), v[b].detach())

        # Cache last state
        if self.stateful:
            self._last_h = h_seq[:, -1, :, :].detach()

        return self.dropout(y_ssm), h_seq

    def _recurrent_fallback(self, A_bar, B_bar, x_ssm, h0, stateful):
        """
        Parallel prefix-scan implementation of:
            h_t = A_bar_t ⊙ h_{t-1} + B_bar_t
        Shapes:
            A_bar, B_bar ∈ (B, T, D, N)
        Output:
            h_seq ∈ (B, T, D, N)
        Behavior identical to sequential version.
        """
        B, T, D, N = A_bar.shape
        dtype = A_bar.dtype
        device = A_bar.device
    
        # --- initial state selection (same logic as before) ---
        if stateful and self._last_h is not None:
            h_prev = self._last_h
        elif h0 is not None:
            h_prev = h0
        else:
            h_prev = torch.zeros(B, D, N, dtype=dtype, device=device)
 
        logA = torch.log(torch.clamp(A_bar, min=1e-6))   # (B,T,D,N)
        log_prefix_prod = torch.cumsum(logA, dim=1)      # (B,T,D,N)
        prefix_prod = torch.exp(log_prefix_prod)         # (B,T,D,N)

        A_shift = torch.cat([torch.ones(B, 1, D, N, device=device, dtype=dtype), A_bar[:, :-1]], dim=1)
        logA_shift = torch.log(torch.clamp(A_shift, min=1e-6))
        log_prefix_prod_shift = torch.cumsum(logA_shift, dim=1)
        prefix_prod_shift = torch.exp(log_prefix_prod_shift)

        weighted_B = B_bar * prefix_prod_shift
        cumsum_weighted = torch.cumsum(weighted_B, dim=1)
        h_seq = cumsum_weighted / prefix_prod_shift 
        h_seq += prefix_prod_shift * h_prev.unsqueeze(1)
        return h_seq

