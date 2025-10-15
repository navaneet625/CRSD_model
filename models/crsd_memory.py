import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 🧠 Dynamic Hebbian Associative Memory
# ============================================================
class HebbianMemory(nn.Module):
    """
    Dynamic Hebbian associative memory 
    - Consistent key/value dimensionality with CRSDCell (d_k, d_v).
    - Supports top-k sparse recall and stable temperature scaling.
    """

    def __init__(self, d_k, d_v, gamma_init=0.999, eta_init=1e-2,
                 topk=None, memory_dtype=torch.float32):
        super().__init__()
        self.memory_dtype = memory_dtype

        # Persist key/value dims
        self.d_k = int(d_k)
        self.d_v = int(d_v)

        # buffer for memory, lazy allocated per-batch
        self.register_buffer("H", None, persistent=False)

        # trainable update parameters (cast to float32 explicitly)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))
        self.eta = nn.Parameter(torch.tensor(float(eta_init), dtype=torch.float32))
        self.temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.topk = topk

    def _allocate_if_needed(self, B, device):
        """Lazy allocation for dynamic batch sizes."""
        if (self.H is None) or (self.H.size(0) != B):
            self.H = torch.zeros(B, self.d_k, self.d_v, dtype=self.memory_dtype, device=device)

    @torch.no_grad()
    def update(self, k, v):
        """
        Hebbian update rule:
            H ← γH + η(k ⊗ v)
        where k: (B, d_k), v: (B, d_v)
        """
        B, d_k = k.shape
        d_v = v.shape[1]
        self._allocate_if_needed(B, k.device)

        gamma = torch.clamp(self.gamma, 0.0, 1.0)
        eta = torch.clamp(self.eta, 0.0, 1.0)
        delta = torch.bmm(k.unsqueeze(-1), v.unsqueeze(1))  # (B, d_k, d_v)
        self.H.mul_(gamma).add_(eta * delta.to(self.H.dtype))

    def recall(self, C, detach_mem=True):
        """
        Batched cosine-normalized recall.
        Args:
            C: (B, N, d_k) or (B, d_k)
        Returns:
            v_hat: (B, N, d_v) or (B, d_v)
        """
        eps = 1e-8
        device = C.device
        squeeze_out = False
        if C.dim() == 2:
            C = C.unsqueeze(1)
            squeeze_out = True

        B, N, d_k = C.shape
        # If memory not allocated or batch mismatch, return zeros with correct d_v
        if self.H is None or self.H.size(0) != B:
            v_hat = torch.zeros(B, N, self.d_v, device=device, dtype=C.dtype)
            return v_hat.squeeze(1) if squeeze_out else v_hat

        H = self.H.detach() if detach_mem else self.H  # (B, d_k, d_v)

        # Normalize across proper axes
        C_norm = F.normalize(C, dim=-1, eps=eps)  # (B, N, d_k)
        H_norm = F.normalize(H, dim=1, eps=eps)   # (B, d_k, d_v)

        # Contract over d_k (the shared dimension) -> produces (B, N, d_v)
        sims = torch.einsum("bnd,bdk->bnk", C_norm, H_norm)  # (B, N, d_v)

        temp = torch.clamp(self.temp, min=0.1)
        alpha = F.softmax(sims / (temp + eps), dim=-1)

        v_hat = torch.einsum("bnk,bdk->bnd", alpha, H)  # (B, N, d_v)

        if self.topk is not None:
            topk_vals, topk_idx = torch.topk(alpha, self.topk, dim=-1)
            mask = torch.zeros_like(alpha)
            mask.scatter_(2, topk_idx, 1.0)
            alpha = alpha * mask
            alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + eps)
            v_hat = torch.einsum("bnk,bdk->bnd", alpha, H)

        return v_hat.squeeze(1) if squeeze_out else v_hat

# ============================================================
# 🧩 Dynamic Episodic Memory Buffer (batched recall only)
# ============================================================
class EpisodicBuffer(nn.Module):
    """
    Priority-based episodic memory.
    Fully dynamic and batched: always processes multiple queries efficiently.
    """

    def __init__(self, slots, d_k, d_v, memory_dtype=torch.float32):
        super().__init__()
        self.slots = int(slots)
        self.memory_dtype = memory_dtype
        self.temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # Persist dims
        self.d_k = int(d_k)
        self.d_v = int(d_v)

        self.register_buffer("keys", None, persistent=False)
        self.register_buffer("vals", None, persistent=False)
        self.register_buffer("importance", None, persistent=False)
        self._batch_idx = None

    def _allocate_if_needed(self, B, device):
        """Allocate memory on first use or batch size change."""
        if (self.keys is None) or (self.keys.size(0) != B):
            self.keys = torch.zeros(B, self.slots, self.d_k, dtype=self.memory_dtype, device=device)
            self.vals = torch.zeros(B, self.slots, self.d_v, dtype=self.memory_dtype, device=device)
            self.importance = torch.zeros(B, self.slots, device=device)
            self._batch_idx = torch.arange(B, device=device)

    @torch.no_grad()
    def write(self, k, v, importance=None):
        """Replace lowest-importance slot per batch."""
        B, d_k = k.shape
        d_v = v.shape[1]
        device = k.device
        self._allocate_if_needed(B, device)

        if importance is None:
            importance = torch.norm(k, dim=-1)

        _, min_idx = torch.min(self.importance, dim=-1)
        idx = self._batch_idx

        self.keys[idx, min_idx] = k.to(self.memory_dtype)
        self.vals[idx, min_idx] = v.to(self.memory_dtype)
        self.importance[idx, min_idx] = importance.to(self.importance.dtype)

    def recall(self, C, temp=None, detach_mem=True):
        """
        Batched cosine-based recall 
        C: (B, N, d_k) or (B, d_k)
        Returns: v_hat (B, N, d_v) or (B, d_v), alpha
        """
        eps = 1e-8
        device = C.device
        if C.dim() == 2:
            C = C.unsqueeze(1)
            squeeze_out = True
        else:
            squeeze_out = False

        B, N, d_k = C.shape
        # if no memory allocated for this batch, return zeros with correct d_v
        if self.keys is None or self.keys.size(0) != B:
            v_hat = torch.zeros(B, N, self.d_v, device=device, dtype=C.dtype)
            return (v_hat.squeeze(1) if squeeze_out else v_hat, None)

        K = self.keys
        V = self.vals
        if detach_mem:
            K, V = K.detach(), V.detach()

        K_norm = F.normalize(K, dim=-1, eps=eps)
        C_norm = F.normalize(C, dim=-1, eps=eps)

        sims = torch.einsum("bnd,bwd->bnw", C_norm, K_norm)
        temperature = (temp or self.temp).to(C.dtype)
        alpha = F.softmax(sims / (temperature + eps), dim=-1)
        V_hat = torch.einsum("bnw,bwd->bnd", alpha, V)

        return (V_hat.squeeze(1) if squeeze_out else V_hat), alpha
