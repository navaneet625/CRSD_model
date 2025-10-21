import torch
import torch.nn as nn
import torch.nn.functional as F

class KCMemory(nn.Module):
    def __init__(self,
                 d_k: int,
                 d_v: int,
                 rank: int,
                 memory_dtype: torch.dtype = torch.float32,
                 kcm_consolidation_rate: float = 0.10,
                 kcm_reduce: str = "sum",
                 debug: bool = False):
        super().__init__()
        self.d_k = int(d_k)
        self.d_v = int(d_v)
        self.rank = int(rank)
        self.dtype = memory_dtype
        self.lam = float(kcm_consolidation_rate)
        self.reduce_fn = kcm_reduce
        self.debug = bool(debug)

        # persistent memory
        self.register_buffer("M", torch.empty(1, rank, d_v), persistent=True)
        nn.init.xavier_uniform_(self.M)

        self.phi_proj = nn.Linear(d_k, rank)

    # ------------------------------------------------------------------
    def batch_update(self, keys: torch.Tensor, values: torch.Tensor,
                     mu: float = 0.02, tau: float = 1.0):
        """Stable memory update with normalization and clipping."""
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)
        B, T, _ = keys.shape
        keys, values = keys.to(self.dtype), values.to(self.dtype)

        phi = F.relu(self.phi_proj(keys))
        phi = F.normalize(phi, dim=-1)
        values = F.normalize(values, dim=-1)

        # normalize aggregated contribution
        delta_M = torch.einsum('btr,btv->rv', phi, values)
        delta_M /= (B * T)  # <-- critical normalization

        M = self.M[0]
        M = (1 - mu) * M + self.lam * delta_M

        # soft normalization to bound magnitude
        M = M / (M.norm(p='fro') + 1e-6)
        self.M.copy_(M.unsqueeze(0))

        if self.debug:
            print(f"[KCM Update] ||M||={M.norm():.4f}")

    # ------------------------------------------------------------------
    def batch_recall(self, keys: torch.Tensor) -> torch.Tensor:
        """Gradient-enabled recall (no detach)."""
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)
        keys = keys.to(self.dtype)

        phi = F.relu(self.phi_proj(keys))
        phi = F.normalize(phi, dim=-1)

        # use live M for differentiable recall
        v_hat = torch.einsum('btr,rv->btv', phi, self.M[0])
        return v_hat.to(keys.dtype)
