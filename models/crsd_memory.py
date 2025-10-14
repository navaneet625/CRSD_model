import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianMemory(nn.Module):
    def __init__(self, B, d_k, d_v, gamma=0.999, eta=1e-2, memory_dtype=torch.float32):
        super().__init__()
        self.gamma = gamma
        self.eta = eta
        self.memory_dtype = memory_dtype
        self.register_buffer("H", torch.zeros(B, d_k, d_v, dtype=self.memory_dtype))

    @torch.no_grad()
    def update(self, k, v):
        # k: (B,d_k), v: (B,d_v)
        # delta may be lower precision; cast to memory dtype before updating
        delta = k.unsqueeze(-1) @ v.unsqueeze(-2)  # (B,d_k,d_v)
        self.H.mul_(self.gamma).add_(self.eta * delta.to(self.H.dtype))

    def recall(self, c):
        # c: (B,d_k), returns (B,d_v)
        # detach to avoid tracking memory in autograd
        return torch.einsum('bkd,bk->bd', self.H.detach(), c)


class EpisodicBuffer(nn.Module):
    def __init__(self, B, slots, d_k, d_v, memory_dtype=torch.float32):
        super().__init__()
        self.slots = int(slots)
        self.memory_dtype = memory_dtype
        # keys/vals stored in memory_dtype
        self.register_buffer("keys", torch.zeros(B, slots, d_k, dtype=self.memory_dtype))
        self.register_buffer("vals", torch.zeros(B, slots, d_v, dtype=self.memory_dtype))
        # pointer per batch (long), placed on same device when module moved
        self.register_buffer("ptr", torch.zeros(B, dtype=torch.long))
        # optional preallocated batch index (created on device when first used)
        self._batch_idx = None

    @torch.no_grad()
    def write(self, k, v):
        # k: (B,d_k), v: (B,d_v)
        B = k.size(0)
        device = self.keys.device

        # create or reuse batch index on correct device
        if (self._batch_idx is None) or (self._batch_idx.size(0) != B) or (self._batch_idx.device != device):
            self._batch_idx = torch.arange(B, device=device)

        batch_idx = self._batch_idx
        idx = self.ptr  # (B,) already on device when module is on device

        # cast inputs to memory dtype and write without extra CPU<->GPU copies
        self.keys[batch_idx, idx] = k.to(dtype=self.keys.dtype)
        self.vals[batch_idx, idx] = v.to(dtype=self.vals.dtype)

        # increment pointer modulo slots
        self.ptr.add_(1)
        self.ptr.remainder_(self.slots)

    def recall(self, c, temp=1.0, window=None):
        # c: (B,d_k)
        if window is not None:
            K = self.keys[:, -window:]  # (B,W,d_k)
            V = self.vals[:, -window:]  # (B,W,d_v)
        else:
            K, V = self.keys, self.vals

        # detach K,V for safe recall (avoid version-counter issues)
        sims = torch.einsum('bwd,bd->bw', K.detach(), c) 
        alpha = F.softmax(sims / temp, dim=-1)
        v_hat = torch.einsum('bw,bwd->bd', alpha, V.detach())
        return v_hat, alpha
