import torch
import torch.nn as nn
from .crsd_block import CRSDBlock
from .crsd_cell import CRSDCell

class CRSDSequence(nn.Module):
    def __init__(self, vocab_size, emb_dim, **cell_kwargs):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.block = CRSDBlock(cell_ctor=CRSDCell,**cell_kwargs)
        self.out = nn.Linear(cell_kwargs["d_h"], vocab_size)

    def forward(self, ids, train_mode=True):
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        x = self.emb_norm(self.emb(ids))
        y = self.block(x, train_mode=train_mode)
        return self.out(y)
