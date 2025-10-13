import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianMemory(nn.Module):
    def __init__(self, B, d_k, d_v, gamma=0.999, eta=1e-2):
        super().__init__()
        self.gamma, self.eta = gamma, eta
        self.register_buffer("H", torch.zeros(B, d_k, d_v))

    @torch.no_grad()
    def update(self, k, v):
        # k: (B,d_k), v: (B,d_v)
        delta = k.unsqueeze(-1) @ v.unsqueeze(-2)  # (B,d_k,d_v)
        self.H.mul_(self.gamma).add_(self.eta * delta)

    def recall(self, c):
        # c: (B,d_k), returns (B,d_v)
        return torch.einsum('bkd,bk->bd', self.H.detach(), c)

class EpisodicBuffer(nn.Module):
    def __init__(self, B, slots, d_k, d_v):
        super().__init__()
        self.slots = slots
        self.register_buffer("keys", torch.zeros(B, slots, d_k))
        self.register_buffer("vals", torch.zeros(B, slots, d_v))
        self.register_buffer("ptr", torch.zeros(B, dtype=torch.long))

    @torch.no_grad()
    def write(self, k, v):
        # k: (B,d_k), v: (B,d_v)
        idx = self.ptr  # (B,)
        self.keys[torch.arange(k.size(0)), idx] = k
        self.vals[torch.arange(v.size(0)), idx] = v
        self.ptr.add_(1).remainder_(self.slots)

    def recall(self, c, temp=1.0, window=None):
        # c: (B,d_k)
        if window is not None:
            # soft window: use last `window` slots
            K = self.keys[:, -window:]  # (B,W,d_k)
            V = self.vals[:, -window:]  # (B,W,d_v)
        else:
            K, V = self.keys, self.vals
        sims = torch.einsum('bwd,bd->bw', K.detach(), c)  # (B,W)
        alpha = F.softmax(sims / temp, dim=-1)           # (B,W)
        v_hat = torch.einsum('bw,bwd->bd', alpha, V.detach())
        return v_hat, alpha
