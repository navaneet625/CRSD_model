import torch
import torch.nn as nn
from .crsd_cell import CRSDCell

class CRSDSequence(nn.Module):
    def __init__(self, vocab_size, emb_dim, **cell_kwargs):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.cell = CRSDCell(d_x=emb_dim, **cell_kwargs)
        self.out = nn.Linear(cell_kwargs['d_h'], vocab_size)

    def forward(self, ids):
        # ids: (T,) or (B, T)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        device = ids.device
        emb = self.emb(ids)  # (B, T, emb_dim) on the same device as ids
        logits = []
        for b in range(B):
            # Initialize hidden state and reservoirs on correct device
            h = torch.zeros(self.cell.d_h, device=device)
            reservoirs = [torch.zeros(d, device=device) for d in self.cell.res_dims]
            for t in range(T):
                x = emb[b, t]   # embedding for batch b at time t
                h, reservoirs = self.cell(x, h, reservoirs)
                logits.append(self.out(h))
        # Stack and reshape to (B, T, vocab_size)
        logits = torch.stack(logits, dim=0).view(B, T, -1)
        return logits
