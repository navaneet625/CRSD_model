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

        # persistent shared memory M ∈ ℝ[1, rank, d_v]
        self.register_buffer("M", torch.empty(1, rank, d_v), persistent=True)
        nn.init.xavier_uniform_(self.M)

        # kernelized feature projection φ(k) = ReLU(Wφ k)
        self.phi_proj = nn.Linear(d_k, rank)

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def parallel_update(self, keys: torch.Tensor, values: torch.Tensor,
                        mu: float = 0.02, tau: float = 5.0):
        """
        Stable memory update using Oja-style normalized Hebbian rule with soft decay.
        keys:  (T, d_k)
        values:(T, d_v)
        """
        keys, values = keys.to(self.dtype), values.to(self.dtype)
        phi = F.relu(self.phi_proj(keys))          # (T, rank)
        phi = phi / (phi.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        values = values / (values.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        delta_M = torch.einsum('tr,tv->rv', phi, values)   # (rank, d_v)
        M = self.M[0]
        M.mul_(1 - mu)                                    # decay (forget old info)
        M.add_(self.lam * delta_M)                        # consolidated write
        norm = M.norm(p='fro')
        if norm > tau:
            M.div_(norm / tau)
        self.M.data.copy_(M.unsqueeze(0))


    # ----------------------------------------------------------------------
    def parallel_recall(self, keys: torch.Tensor) -> torch.Tensor:
        keys = keys.to(self.dtype)
        phi = F.relu(self.phi_proj(keys))                  # (T, rank)
        v_hat = torch.einsum('tr,rv->tv', phi, self.M[0])  # (T, d_v)
        return v_hat.to(keys.dtype)

    # ----------------------------------------------------------------------
    def causal_sequence(self, keys: torch.Tensor, values: torch.Tensor):
        lam = self.lam
        alpha = 1.0 - lam
        T = keys.shape[0]
        device = keys.device

        phi = F.relu(self.phi_proj(keys))                   # (T, rank)
        S = torch.einsum('tr,tv->trv', phi, values)         # (T, rank, d_v)

        # prefix-scan trick: Z_t = M_t / α^t  ⇒  Z_t = M0 + λ Σ (S_i / α^i)
        t = torch.arange(1, T + 1, device=device, dtype=keys.dtype)
        a_pos = alpha ** t
        a_neg = alpha ** (-t)

        S_scaled = S * a_neg.view(T, 1, 1)
        Z = self.M + lam * torch.cumsum(S_scaled, dim=0)
        M_seq = Z * a_pos.view(T, 1, 1)

        v_hat = torch.einsum('tr,trv->tv', phi, M_seq)
        return M_seq, v_hat
