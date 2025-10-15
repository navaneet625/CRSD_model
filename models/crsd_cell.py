import torch
import torch.nn as nn
import torch.nn.functional as F
from .crsd_memory import HebbianMemory, EpisodicBuffer


class CRSDCell(nn.Module):
    """
    - Fully vectorized, FFT-based sequence processing.
    - Uses Hebbian + Episodic memory once per sequence (batched recall/write).
    - Ideal for inference or fine-tuning.
    """

    def __init__(self, d_x, d_h, res_dims, d_k, d_v,
                 mem_slots=256,
                 hebbian_topk: int | None = 16,
                 episodic_topk: int | None = 8,
                 auto_write: bool = False,
                 use_diag_A: bool = True,
                 memory_dtype=torch.float32,
                 res_dropout_p: float = 0.0,
                 init_alpha_bias: float = -1.0):
        super().__init__()
        self.d_x = d_x
        self.d_h = d_h
        self.res_dims = list(res_dims)
        self.R = int(sum(self.res_dims))

        # -------- Reservoir projections --------
        self.Wx = nn.Linear(d_x, self.R)
        self.Wh = nn.Linear(d_h, self.R)
        self.res_logit_alpha = nn.Parameter(torch.full((len(res_dims),), float(init_alpha_bias)))

        # -------- Hidden transforms --------
        self.use_diag_A = use_diag_A
        if use_diag_A:
            self.A_scale = nn.Parameter(torch.ones(d_h))
        else:
            self.A = nn.Linear(d_h, d_h, bias=False)

        self.B = nn.Linear(self.R, d_h, bias=False)
        self.U = nn.Linear(d_x, d_h)

        # -------- FFT spectral mixer --------
        self.fft_norm = nn.LayerNorm(d_h)
        self.fft_proj = nn.Linear(d_h, d_h)
        self.fft_mix = nn.Parameter(torch.tensor(0.5))

        # -------- Memory interface --------
        self.key_in = nn.Linear(self.R + d_x, d_k)
        self.val_in = nn.Linear(d_h, d_v)
        self.recall_map = nn.Linear(d_v, d_h)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        self._heb = None
        self._buf = None
        self.memory_dtype = memory_dtype
        self.mem_slots = mem_slots
        self.hebbian_topk = hebbian_topk
        self.episodic_topk = episodic_topk
        self.auto_write = auto_write

        # -------- Normalization + dropout --------
        self.r_norm = nn.LayerNorm(self.R)
        self.res_dropout = nn.Dropout(res_dropout_p) if res_dropout_p > 0 else nn.Identity()

        # -------- Segmentation index --------
        seg_sizes = torch.tensor(self.res_dims, dtype=torch.long)
        seg_ids = torch.arange(len(self.res_dims), dtype=torch.long)
        self.register_buffer("seg_index", torch.repeat_interleave(seg_ids, seg_sizes), persistent=False)

        # -------- FFT cache --------
        self.register_buffer("_fft_n", torch.tensor(self.d_h, dtype=torch.long), persistent=False)

        # -------- Init weights --------
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.75 if (m is self.Wx or m is self.Wh) else 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _ensure_mem(self, B, d_k, d_v, device):
        """
        Ensure HebbianMemory and EpisodicBuffer exist and match batch size B.
        Safe against None buffers or uninitialized tensors.
        """
        def _need_alloc(mem_obj, attr_name="H"):
            """Check if memory object exists and has correct batch dimension."""
            if mem_obj is None:
                return True
            attr = getattr(mem_obj, attr_name, None)
            if attr is None:
                return True
            try:
                return attr.size(0) != B
            except Exception:
                return True
    
        heb_needs_alloc = _need_alloc(self._heb, "H")
        buf_needs_alloc = _need_alloc(self._buf, "keys")
    
        if heb_needs_alloc or buf_needs_alloc:
            # Create new fresh memories
            self._heb = HebbianMemory(d_k, d_v, memory_dtype=self.memory_dtype).to(device)
            self._buf = EpisodicBuffer(self.mem_slots, d_k, d_v, memory_dtype=self.memory_dtype).to(device)
    
         
    @torch.no_grad()
    def write_memory(self, k, v, importance=None):
        self._buf.write(k, v, importance)
        self._heb.update(k, v)

    def forward(self, x_seq, h0=None, r0=None, train_mode=True):
        """
        Parallel forward (FFT + memory mixing)
        -------------------------------------
        Args:
            x_seq: (B, T, d_x)
        Returns:
            hs: (B, T, d_h) sequence of hidden states
            rs: (B, T, R)  reservoir states
        """
        B, T, _ = x_seq.shape
        device = x_seq.device
        self._ensure_mem(B, self.key_in.out_features, self.val_in.out_features, device)

        # initial states
        h_prev = torch.zeros(B, self.d_h, device=device) if h0 is None else h0
        r_prev = torch.zeros(B, self.R, device=device) if r0 is None else r0

        # --- Reservoir projection (vectorized) ---
        Wx = self.Wx(x_seq)  # (B, T, R)
        Wh = torch.matmul(h_prev.unsqueeze(1), self.Wh.weight.T).expand(B, T, self.R)
        alpha = torch.sigmoid(self.res_logit_alpha[self.seg_index]).to(device)
        alpha = alpha.unsqueeze(0).unsqueeze(1).expand(B, T, -1)
        r_seq = alpha * r_prev.unsqueeze(1) + (1 - alpha) * torch.tanh(Wx + Wh)
        r_seq = self.r_norm(r_seq)
        r_seq = self.res_dropout(r_seq)

        # --- Hidden transform ---
        Ah = self.A_scale * h_prev.unsqueeze(1)
        Uh = self.U(x_seq)
        Bh = torch.matmul(r_seq, self.B.weight.T)
        h_core = Ah + Bh + Uh
        h_tilde = F.gelu(h_core)

        # --- FFT spectral mixing (global context) ---
        h_norm = self.fft_norm(h_tilde)
        # n = int(self._fft_n.item())
        n = self.d_h
        h_freq = torch.fft.rfft(h_norm, n=n, dim=-1, norm='ortho')
        mix_scale = torch.sigmoid(self.fft_mix)
        h_ifft = torch.fft.irfft(h_freq * mix_scale, n=n, dim=-1, norm='ortho')
        h_fft_out = self.fft_proj(h_ifft)
        s = torch.sigmoid(self.fft_mix)
        hs = (1 - s) * h_tilde + s * h_fft_out

        # --- Memory recall (batched once per sequence) ---
        k = self.key_in(torch.cat([r_seq, x_seq], dim=-1))  # (B, T, d_k)
        v = self.val_in(hs.detach() if not train_mode else hs)  # (B, T, d_v)
        c = F.normalize(k, dim=-1, eps=1e-8)

        # EpisodicBuffer.recall returns (v_hat, alpha); HebbianMemory.recall returns v_hat
        v_hat, _ = self._buf.recall(c.mean(dim=1))  # -> (B, d_v)
        v_hebb = self._heb.recall(c.mean(dim=1))   # -> (B, d_v)

        # Ensure tensors exist and are 2D (B, d_v)
        if v_hat is None:
            v_hat = torch.zeros(B, self.val_in.out_features, device=device, dtype=v.dtype)
        if v_hebb is None:
            v_hebb = torch.zeros(B, self.val_in.out_features, device=device, dtype=v.dtype)

        # pad or truncate helper
        target_dv = self.val_in.out_features
        def _pad_or_trunc(x, target):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            dv = x.size(-1)
            if dv == target:
                return x
            if dv < target:
                pad = torch.zeros(*x.shape[:-1], target - dv, device=x.device, dtype=x.dtype)
                return torch.cat([x, pad], dim=-1)
            # dv > target: truncate
            return x[..., :target]

        v_hat = _pad_or_trunc(v_hat, target_dv)
        v_hebb = _pad_or_trunc(v_hebb, target_dv)

        mix = torch.sigmoid(self.mix_logit)
        v_comb = mix * v_hat + (1.0 - mix) * v_hebb

        # ensure final dim matches recall_map
        if v_comb.size(-1) != self.recall_map.in_features:
            v_comb = _pad_or_trunc(v_comb, self.recall_map.in_features)

        h_mem = self.recall_map(v_comb).unsqueeze(1)
        hs = hs + 0.5 * h_mem  # small additive memory fusion

        # --- Memory write (optional) ---
        if self.auto_write and train_mode:
            with torch.no_grad():
                k_last = k[:, -1, :].detach()
                v_last = v[:, -1, :].detach()
                self.write_memory(k_last, v_last)

        return hs, r_seq
