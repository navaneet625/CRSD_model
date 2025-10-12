import torch

class HebbianMemory:
    def __init__(self, d_k, d_v, gamma=0.999, eta=1e-2, device='cpu'):
        self.d_k, self.d_v = d_k, d_v
        self.gamma = gamma
        self.eta = eta
        self.device = device
        self.H = torch.zeros(d_k, d_v, device=device)

    def update(self, k, v):
        # k: (d_k,), v: (d_v,)
        delta = k.unsqueeze(1) @ v.unsqueeze(0)
        self.H = self.gamma * self.H + self.eta * delta

    def recall(self, c):
        # returns v_hebb = H^T c
        return self.H.t() @ c

class EpisodicBuffer:
    def __init__(self, slots, d_k, d_v, device='cpu'):
        self.slots = slots
        self.d_k = d_k
        self.d_v = d_v
        self.device = device
        self.keys = torch.zeros(slots, d_k, device=device)
        self.vals = torch.zeros(slots, d_v, device=device)
        self.ptr = 0

    def write(self, k, v):
        self.keys[self.ptr].copy_(k)
        self.vals[self.ptr].copy_(v)
        self.ptr = (self.ptr + 1) % self.slots

    def recall(self, c, temp=1.0):
        sims = self.keys @ c
        alpha = torch.softmax(sims / temp, dim=0)
        return (alpha.unsqueeze(0) @ self.vals).squeeze(0), alpha