import torch
import torch.nn as nn
import torch.nn.functional as F
from .crsd_memory import HebbianMemory, EpisodicBuffer

class CRSDCell(nn.Module):
    def __init__(self, d_x, d_h, res_dims, d_k, d_v, mem_slots=64,
                 mem_update_every=1, mem_read_window=None):
        super().__init__()
        self.d_x, self.d_h = d_x, d_h
        self.res_dims = res_dims
        self.R = int(sum(res_dims))

        # single big reservoir projections
        self.Wx = nn.Linear(d_x, self.R, bias=True)
        self.Wh = nn.Linear(d_h, self.R, bias=True)
        self.res_logit_alpha = nn.Parameter(torch.randn(len(res_dims)))  # (S,)

        # base transforms
        self.A = nn.Linear(d_h, d_h, bias=False)
        self.B = nn.Linear(self.R, d_h, bias=False)
        self.U = nn.Linear(d_x, d_h, bias=True)

        # proj & gate (fused)
        self.key_in = nn.Linear(self.R + d_x, d_k, bias=True)
        self.val_in = nn.Linear(d_h, d_v, bias=True)
        self.gate_in = nn.Linear(d_h + self.R + d_k + d_v, d_h, bias=True)
        self.recall_map = nn.Linear(d_v, d_h, bias=True)

        # runtime knobs
        self.mem_update_every = int(max(1, mem_update_every))
        self.mem_read_window = mem_read_window
        self.mem_slots = int(mem_slots)

        self._heb = None
        self._buf = None

        # ---- init (safe for layers w/o bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ---- precompute a segment index to vectorize alpha (no .item())
        seg_sizes = torch.tensor(res_dims, dtype=torch.long)
        seg_ids = torch.arange(len(res_dims), dtype=torch.long)
        seg_index = torch.repeat_interleave(seg_ids, seg_sizes)   # (R,)
        self.register_buffer("seg_index", seg_index, persistent=False)

    def _alpha_full(self, B, device):
        # alpha per segment → broadcast to (R,) using seg_index → expand to (B,R)
        alphas = torch.sigmoid(self.res_logit_alpha)              # (S,)
        alpha_vec = alphas[self.seg_index].to(device)             # (R,)
        return alpha_vec.unsqueeze(0).expand(B, -1)               # (B,R)

    def _ensure_mem(self, B, d_k, d_v, device):
        # create or recreate memories when batch size changes
        if (self._heb is None) or (self._heb.H.size(0) != B):
            self._heb = HebbianMemory(B, d_k, d_v, memory_dtype=self._buf_dtype() if hasattr(self, "_buf_dtype") else torch.float32).to(device)
            self._buf = EpisodicBuffer(B, slots=self.mem_slots, d_k=d_k, d_v=d_v, memory_dtype=self._heb.H.dtype).to(device)

    def _buf_dtype(self):
        # choose memory dtype (fallback to float32)
        return torch.float32

    def forward(self, x_t, h_prev, r_prev, step, train_mode=True):
        # x_t: (B,d_x), h_prev: (B,d_h), r_prev: (B,R)
        B = x_t.size(0)
        device = x_t.device

        alpha = self._alpha_full(B, device)                       # (B,R)

        # ----- reservoir update (NO in-place)
        inp = self.Wx(x_t) + self.Wh(h_prev)                      # (B,R)
        inp = torch.clamp(inp, -5.0, 5.0)
        r_new = alpha * r_prev + (1.0 - alpha) * torch.tanh(inp)
        r_new = torch.clamp(r_new, -5.0, 5.0)

        # ----- base hidden (NO in-place)
        h_core = self.A(h_prev) + self.B(r_new) + self.U(x_t)
        h_tilde = torch.clamp(F.gelu(h_core), -5.0, 5.0)

        # ----- kv
        k = self.key_in(torch.cat([r_new, x_t], dim=-1))          # (B,d_k)
        v = self.val_in(h_tilde.detach())                         # (B,d_v)  (stop grad into memory)

        # Ensure memories exist (and are on correct device)
        self._ensure_mem(B, k.size(-1), v.size(-1), device)

        # ======== RECALL FIRST FROM DETACHED BUFFERS (no expensive clone) ========
        K_snap = self._buf.keys.detach()    # (B,slots,d_k)
        V_snap = self._buf.vals.detach()    # (B,slots,d_v)
        H_snap = self._heb.H.detach()       # (B,d_k,d_v)

        c = F.normalize(k, dim=-1)                                # (B,d_k)

        # episodic recall (optionally windowed)
        if self.mem_read_window is not None:
            Kw = K_snap[:, -self.mem_read_window:, :]            # (B,W,d_k)
            Vw = V_snap[:, -self.mem_read_window:, :]            # (B,W,d_v)
        else:
            Kw, Vw = K_snap, V_snap

        sims = torch.einsum('bwd,bd->bw', Kw, c)                  # (B,W)
        alpha_att = F.softmax(sims, dim=-1)                       # (B,W)
        v_buf = torch.einsum('bw,bwd->bd', alpha_att, Vw)         # (B,d_v)

        # hebbian recall
        v_hebb = torch.einsum('bkd,bk->bd', H_snap, c)            # (B,d_v)

        v_hat = 0.5 * v_buf + 0.5 * v_hebb                        # (B,d_v)

        # gating and fusion
        g_in = torch.cat([h_prev, r_new, c, v_hat], dim=-1)
        g = torch.sigmoid(self.gate_in(g_in))
        chi = self.recall_map(v_hat)
        h_new = torch.clamp((1.0 - g) * h_tilde + g * chi, -5.0, 5.0)

        # ======== THEN UPDATE LIVE MEMORIES (after recall) ========
        if train_mode and (step % self.mem_update_every == 0):
            with torch.no_grad():
                # write detached k and v (avoid changing their requires_grad)
                self._buf.write(k.detach(), v.detach())
                self._heb.update(k.detach(), v.detach())

        return h_new, r_new
