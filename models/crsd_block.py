import torch
import torch.nn as nn
from .crsd_cell import CRSDCell
from .crsd_kcm import KCMemory


class CRSDBlock(nn.Module):
    """
    Multi-layer CRSDSSM (Parallel State Space Model + shared KCM) block.
    Handles d_x -> d_h projection for the first layer.
    """

    def __init__(self,d_k: int,d_v: int,rank_kcm: int,kcm_consolidation_rate: float,kcm_reduce: str,use_rope: bool,auto_write: bool,
        res_dropout_p: float,
        memory_dtype: torch.dtype,
        cell_ctor=CRSDCell,
        layers: int = 2,
        d_x: int | None = None,
        d_h: int | None = None,
        ssm_N: int | None = None,
        shared_mem: KCMemory | None = None,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.d_x = d_x
        self.d_h = d_h

        self.input_proj = nn.Identity()
        if d_x != d_h:
            print(f"CRSDBlock: Projecting input from d_x={d_x} to d_h={d_h}")
            self.input_proj = nn.Linear(d_x, d_h)

        if shared_mem is None:
            print("Warning: CRSDBlock initialized without a shared KCMemory instance.")
            shared_mem = KCMemory(d_k=d_k, d_v=d_v, rank=rank_kcm)

        # FIX 2: Clean argument forwarding
        clean_cell_kwargs = {
            "d_k": d_k,
            "d_v": d_v,
            "rank_kcm": rank_kcm,
            "kcm_consolidation_rate": kcm_consolidation_rate,
            "kcm_reduce": kcm_reduce,
            "use_rope": use_rope,
            "auto_write": auto_write,
            "res_dropout_p": res_dropout_p,
            "memory_dtype": memory_dtype,
        }
        for i in range(layers):
            self.layers.append(cell_ctor(d_x=d_h,d_h=d_h,N=ssm_N,shared_mem=shared_mem,**clean_cell_kwargs,))

    def forward(
        self,
        x_seq: torch.Tensor,
        train_mode: bool = True,
        rope_cache: tuple | None = None,
        h0_list: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """
        Forward pass through all stacked CRSDCells.
        """
        h_seq = self.input_proj(x_seq)
        h_next_list = []

        if h0_list is None:
            h0_list = [None] * len(self.layers)

        for i, cell in enumerate(self.layers):
            cell.train(mode=train_mode)
            h0 = h0_list[i]

            h_seq, h_state_seq = cell(
                h_seq,
                h0=h0,
                train_mode=train_mode,
                rope_cache=rope_cache,
            )
            h_next_list.append(h_state_seq[:, -1, :, :].detach() if h_state_seq is not None else None)

        return h_seq, h_next_list
