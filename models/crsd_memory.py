import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianMemory(nn.Module):
    def __init__(self, d_k, d_v, gamma=0.999, eta=1e-2):
        super().__init__()
        self.gamma = gamma
        self.eta = eta
        # buffer → moves with .to(), not trainable
        self.register_buffer("H", torch.zeros(d_k, d_v))

    @torch.no_grad()
    def update(self, k, v):
        # k: (d_k,), v: (d_v,)
        delta = k.unsqueeze(1) @ v.unsqueeze(0)  # (d_k, d_v)
        self.H.mul_(self.gamma).add_(self.eta * delta)

    def recall(self, c):
        # ⚠️ Detach + CLONE to avoid sharing storage with a tensor we will mutate later
        H_safe = self.H.detach().clone()     # (d_k, d_v)
        return H_safe.t() @ c                # (d_v,)

class EpisodicBuffer(nn.Module):
    def __init__(self, slots, d_k, d_v):
        super().__init__()
        self.slots = slots
        self.register_buffer("keys", torch.zeros(slots, d_k))
        self.register_buffer("vals", torch.zeros(slots, d_v))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def write(self, k, v):
        idx = int(self.ptr.item())
        self.keys[idx].copy_(k)
        self.vals[idx].copy_(v)
        self.ptr.copy_(((self.ptr + 1) % self.slots))

    def recall(self, c, temp=1.0):
        # ⚠️ Detach + CLONE to break storage aliasing with later writes in the same forward
        keys = self.keys.detach().clone()    # (slots, d_k)
        vals = self.vals.detach().clone()    # (slots, d_v)
        sims = keys @ c                      # (slots,)
        alpha = F.softmax(sims / temp, dim=0)
        v_hat = (alpha.unsqueeze(0) @ vals).squeeze(0)
        return v_hat, alpha
