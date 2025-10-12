import torch
import torch.nn as nn
from .crsd_cell import CRSDCell

class CRSDBlock(nn.Module):
    def __init__(self, cell_ctor, layers=1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([cell_ctor(**kwargs) for _ in range(layers)])

    def forward(self, x_seq):
        # x_seq: (T, d_x)
        T = x_seq.shape[0]
        for cell in self.layers:
            h = x_seq.new_zeros(cell.d_h)
            reservoirs = [x_seq.new_zeros(d) for d in cell.res_dims]
            out_seq = []
            for t in range(T):
                h, reservoirs = cell(x_seq[t], h, reservoirs)
                out_seq.append(h)
            x_seq = torch.stack(out_seq, dim=0)
        return x_seq
