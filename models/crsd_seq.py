import torch
import torch.nn as nn
from .crsd_block import CRSDBlock
from .crsd_cell import CRSDCell

class CRSDSequence(nn.Module):
    """
    Embedding → optional projection → CRSDBlock → linear output.
    """
    def __init__(self, vocab_size, emb_dim, d_x, **cell_kwargs):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)

        # Project embedding to d_x if needed
        self.emb_proj = nn.Identity() if emb_dim == d_x else nn.Linear(emb_dim, d_x)

        # Remove irrelevant kwargs
        cell_kwargs.pop("emb_dim", None)
        cell_kwargs.pop("d_x", None)

        # Core recurrent block
        self.block = CRSDBlock(cell_ctor=CRSDCell, d_x=d_x, **cell_kwargs)

        # Output projection
        self.out = nn.Linear(cell_kwargs["d_h"], vocab_size)

    def forward(self, ids, train_mode=True):
        """
        Forward pass for token IDs:
        ids: (B, T) or (T,) long tensor
        Returns logits: (B, T, vocab_size)
        """
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)  # (1,T)

        x = self.emb(ids)        # (B,T,emb_dim)
        x = self.emb_norm(x)
        x = self.emb_proj(x)     # (B,T,d_x)

        # Pass through recurrent block
        y = self.block(x, train_mode=train_mode)  # (B,T,d_h)

        # Project to vocab logits
        logits = self.out(y)     # (B,T,vocab_size)
        return logits
