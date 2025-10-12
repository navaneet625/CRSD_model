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
        emb = self.emb(ids)  # (B,T,emb)
        logits = []
        for b in range(B):
            h = torch.zeros(self.cell.d_h)
            reservoirs = [torch.zeros(d) for d in self.cell.res_dims]
            for t in range(T):
                x = emb[b,t]
                h, reservoirs = self.cell(x, h, reservoirs)
                logits.append(self.out(h))
        logits = torch.stack(logits, dim=0)
        return logits