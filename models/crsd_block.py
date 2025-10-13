import torch
import torch.nn as nn
from .crsd_cell import CRSDCell

class CRSDBlock(nn.Module):
    """
    Batched (B,T,*) interface. Keeps loop over time (recurrent),
    but fuses per-step ops and avoids Python loops over reservoirs.
    """
    def __init__(self, cell_ctor=CRSDCell, layers=1, d_x=None, d_h=None, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            cell_ctor(d_x=d_x, d_h=d_h, **kwargs) for _ in range(layers)
        ])
    def forward(self, x_seq, train_mode=True):
        # x_seq: (B,T,d_x)
        B, T, d_x = x_seq.shape
        h = None
        r = None

        for layer, cell in enumerate(self.layers):
            # init states per layer
            if h is None:
                d_h = cell.d_h
                h = x_seq.new_zeros(B, d_h)
            else:
                h = h.detach().zero_()  # reinit per layer

            r = x_seq.new_zeros(B, cell.R)

            outs = []
            for t in range(T):
                h, r = cell(x_seq[:, t, :], h, r, step=t, train_mode=train_mode)
                outs.append(h)

            x_seq = torch.stack(outs, dim=1)  # (B,T,d_h)
        return x_seq
