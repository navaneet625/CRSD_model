import torch
import torch.nn as nn
from .crsd_kcm import KCMemory 
from .crsd_block import CRSDBlock
from .crsd_cell import CRSDCell 


def _generate_rope_cache(T: int, D_model: int, device: torch.device):
    """Generates shared sin/cos cache for RoPE."""
    D_rope = D_model // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_rope, 2, device=device).float() / D_rope))
    t = torch.arange(T, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
    sin, cos = emb.sin(), emb.cos()
    return sin, cos


class CRSDSequence(nn.Module):
    """
    CRSD Sequence Model: Embedding -> stacked CRSDCells -> Output.
    Handles shared KCMemory, RoPE cache, and state caching.
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        d_x: int,
        d_h: int,
        ssm_N: int, 
        d_k: int,
        d_v: int,
        rank_kcm: int = 64,
        use_rope: bool = True,
        auto_write: bool = True,
        res_dropout_p: float = 0.0,
        use_diag_A: bool = True, 
        memory_dtype: torch.dtype = torch.float32,
        layers: int = 2,
        kcm_consolidation_rate: float = 0.10,
        kcm_reduce: str = "sum",
        max_len: int = 2048,
    ):
        super().__init__()
        self.d_h = int(d_h)
        self.layers = int(layers)
        self.max_len = int(max_len)
        self.use_rope = bool(use_rope)

        # --- Embedding ---
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.emb_proj = nn.Identity() if emb_dim == d_x else nn.Linear(emb_dim, d_x)

        # --- Shared Kernelized Continuous Memory ---
        self.kcm = KCMemory(
            d_k=d_k, d_v=d_v, rank=rank_kcm, memory_dtype=memory_dtype,
            kcm_consolidation_rate=kcm_consolidation_rate,
            kcm_reduce=kcm_reduce,
        )

        # --- Core stacked CRSD SSM block ---
        block_kwargs = {
            "d_k": d_k,
            "d_v": d_v,
            "rank_kcm": rank_kcm,
            "kcm_consolidation_rate": kcm_consolidation_rate,
            "kcm_reduce": kcm_reduce,
            "use_rope": use_rope,
            "auto_write": auto_write,
            "res_dropout_p": res_dropout_p,
            "memory_dtype": memory_dtype,
            "shared_mem": self.kcm,
        }
        self.block = CRSDBlock(
            cell_ctor=CRSDCell,
            layers=layers,
            d_x=d_x,
            d_h=d_h,
            ssm_N=ssm_N,
            **block_kwargs,
        )

        # --- Output projection ---
        self.out = nn.Linear(d_h, vocab_size)

        # --- Cached inference states ---
        self._rope_cache = {}
        self._last_ssm_states = [None] * layers

    # ------------------------------------------------------------------
    def forward(self, ids: torch.Tensor, train_mode: bool = True, rope_cache=None):
        """
        ids: (B, T) integer token IDs
        Returns:
            if train_mode: (logits, k, v)
            else: logits
        """
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        device = ids.device

        # --- RoPE Cache ---
        if self.use_rope and rope_cache is None and T <= self.max_len:
            if T not in self._rope_cache or self._rope_cache[T][0].device != device:
                self._rope_cache[T] = _generate_rope_cache(T, self.d_h, device)
            rope_cache = self._rope_cache[T]

        # --- Initial states ---
        initial_states = self._last_ssm_states if not train_mode else None

        # --- Embedding + projection ---
        x = self.emb(ids)
        x = self.emb_norm(x)
        x = self.emb_proj(x)

        # --- Pass through CRSD block ---
        if train_mode:
            # Each CRSDCell now returns (y, h_seq, k, v)
            h_seq, next_states, k_all, v_all = self.block(
                x, train_mode=True, rope_cache=rope_cache, h0_list=initial_states
            )
        else:
            h_seq, next_states = self.block(
                x, train_mode=False, rope_cache=rope_cache, h0_list=initial_states
            )

        if not train_mode:
            self._last_ssm_states = next_states

        # --- Output projection ---
        logits = self.out(h_seq)  # (B, T, vocab_size)

        if train_mode:
            if isinstance(k_all, (list, tuple)):
                k, v = k_all[-1], v_all[-1]
            else:
                k, v = k_all, v_all
            return logits, k, v
        else:
            return logits, None, None

