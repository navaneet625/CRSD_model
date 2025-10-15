import torch
import torch.nn as nn
from .crsd_cell import CRSDCell

class CRSDBlock(nn.Module):
    """
    - Processes entire sequences per layer in parallel
    - Compatible with torch.compile, JIT, and AMP
    """

    def __init__(self, cell_ctor=CRSDCell, layers=6, d_x=None, d_h=None, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            in_dim = d_x if i == 0 else d_h
            self.layers.append(cell_ctor(d_x=in_dim, d_h=d_h, **kwargs))

    def forward(self, x_seq: torch.Tensor, train_mode: bool = True) -> torch.Tensor:
        """
        Args:
            x_seq (Tensor): (B, T, d_x)
            train_mode (bool): controls memory recall/write and dropout behavior
        Returns:
            Tensor: (B, T, d_h)
        """
        for i, cell in enumerate(self.layers):
            cell.train(mode=train_mode)
            # call the parallel forward (FFT + memory) per layer
            h_out, _ = cell(x_seq, train_mode=train_mode)
            x_seq = h_out

        return x_seq
