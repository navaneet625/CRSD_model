import torch
import torch.nn as nn
import torch.nn.functional as F
from .crsd_memory import HebbianMemory, EpisodicBuffer

class CRSDCell(nn.Module):
    def __init__(self, d_x, d_h, res_dims, d_k, d_v, mem_slots=64, device='cpu'):
        super().__init__()
        self.d_x = d_x
        self.d_h = d_h
        self.res_dims = res_dims
        self.total_res = sum(res_dims)
        self.d_k = d_k
        self.d_v = d_v
        self.device = device

        # reservoirs
        self.res_Wx = nn.ModuleList([nn.Linear(d_x, d) for d in res_dims])
        self.res_Wh = nn.ModuleList([nn.Linear(d_h, d) for d in res_dims])
        self.res_logit_alpha = nn.Parameter(torch.randn(len(res_dims)))

        # base
        self.A = nn.Linear(d_h, d_h, bias=False)
        self.B = nn.Linear(self.total_res, d_h, bias=False)
        self.U = nn.Linear(d_x, d_h)

        # projections
        self.key_proj = nn.Linear(self.total_res + d_x, d_k)
        self.val_proj = nn.Linear(d_h, d_v)

        # gating
        self.gate = nn.Linear(d_h + self.total_res + d_k + d_v, d_h)
        self.recall_map = nn.Linear(d_v, d_h)

        # memory modules
        self.hebb = HebbianMemory(d_k, d_v, device=device)
        self.buffer = EpisodicBuffer(mem_slots, d_k, d_v, device=device)

    def forward(self, x, h_prev, reservoirs):
        # reservoirs: list of tensors
        res_outs = []
        for i,(r_prev, Wx, Wh) in enumerate(zip(reservoirs, self.res_Wx, self.res_Wh)):
            alpha = torch.sigmoid(self.res_logit_alpha[i])
            inp = Wx(x) + Wh(h_prev)
            r_new = alpha * r_prev + (1 - alpha) * torch.tanh(inp)
            res_outs.append(r_new)
        r_cat = torch.cat(res_outs, dim=-1)

        h_tilde = F.gelu(self.A(h_prev) + self.B(r_cat) + self.U(x))

        k = self.key_proj(torch.cat([r_cat, x], dim=-1))
        v = self.val_proj(h_tilde.detach())

        # write
        self.buffer.write(k.detach(), v.detach())
        self.hebb.update(k.detach(), v.detach())

        # recall
        c = F.normalize(k, dim=-1)
        v_buf, alpha = self.buffer.recall(c)
        v_hebb = self.hebb.recall(c)
        v_hat = 0.5 * v_buf + 0.5 * v_hebb

        gate_in = torch.cat([h_prev, r_cat, c, v_hat], dim=-1)
        g = torch.sigmoid(self.gate(gate_in))
        chi = self.recall_map(v_hat)
        h_new = (1 - g) * h_tilde + g * chi

        return h_new, res_outs