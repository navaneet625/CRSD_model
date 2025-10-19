import torch
import torch.nn as nn
from .crsd_kcm import KCMemory 
from .crsd_block import CRSDBlock
from .crsd_cell import CRSDCell 


def _generate_rope_cache(T: int, D_model: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates the shared sin/cos cache for RoPE."""
    D_rope = D_model // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_rope, 2, device=device).float() / D_rope))
    t = torch.arange(T, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) 
    sin = emb.sin()
    cos = emb.cos()
    return sin, cos 


class CRSDSequence(nn.Module):
    """
    CRSD Sequence Model: Embedding -> SSM Block -> Output.
    Manages KCMemory, RoPE cache, and the KCM commitment process.
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
        
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.emb_proj = nn.Identity() if emb_dim == d_x else nn.Linear(emb_dim, d_x)

        self.kcm = KCMemory(
            d_k=d_k, d_v=d_v, rank=rank_kcm, memory_dtype=memory_dtype,
            kcm_consolidation_rate=kcm_consolidation_rate,
            kcm_reduce=kcm_reduce,
        )
        
        # 3. Core CRSDSSM block
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
            **block_kwargs
        )

        # 4. Final output projection
        self.out = nn.Linear(d_h, vocab_size)

        # Cached states for stateful inference and RoPE
        self._rope_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._last_ssm_states: list[torch.Tensor | None] = [None] * layers

    def forward(self, ids: torch.Tensor, train_mode: bool = True, rope_cache=None) -> torch.Tensor:
        """
        Forward pass for token IDs using the CRSDSSM architecture.
        ids: (B, T) or (T,)
        Returns: logits: (B, T, vocab_size)
        """
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        device = ids.device

        # --- RoPE Cache Generation/Lookup ---
        if self.use_rope and rope_cache is None and T <= self.max_len:
            if T not in self._rope_cache or self._rope_cache[T][0].device != device:
                self._rope_cache[T] = _generate_rope_cache(T, self.d_h, device)
            rope_cache = self._rope_cache[T]

        # --- Prepare Initial States ---
        initial_states = self._last_ssm_states if not train_mode else None

        # --- Forward Pass ---
        x = self.emb(ids)
        x = self.emb_norm(x)
        x = self.emb_proj(x)
        
        # h_seq: (B, T, D), next_states: list of (B, D, N) SSM states
        h_seq, next_states = self.block(
            x,
            train_mode=train_mode,
            rope_cache=rope_cache, 
            h0_list=initial_states
        )
        
        # --- Cache Final States for Inference ---
        if not train_mode:
            self._last_ssm_states = next_states

        # --- Output Projection ---
        logits = self.out(h_seq)
        return logits
