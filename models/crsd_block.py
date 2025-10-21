import torch
import torch.nn as nn
from .crsd_cell import CRSDCell
from .crsd_kcm import KCMemory


class CRSDBlock(nn.Module):
    """
    Multi-layer CRSDSSM (Parallel State Space Model + shared KCM) block.
    Handles d_x -> d_h projection for the first layer and aggregates keys/values.
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        rank_kcm: int,
        kcm_consolidation_rate: float,
        kcm_reduce: str,
        use_rope: bool,
        auto_write: bool,
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

        # Optional projection if d_x != d_h
        self.input_proj = nn.Identity()
        if d_x != d_h:
            print(f"CRSDBlock: Projecting input from d_x={d_x} to d_h={d_h}")
            self.input_proj = nn.Linear(d_x, d_h)

        # Ensure a shared KCMemory exists
        if shared_mem is None:
            print("⚠️ Warning: CRSDBlock initialized without a shared KCMemory instance.")
            shared_mem = KCMemory(d_k=d_k, d_v=d_v, rank=rank_kcm)

        # Common arguments for each CRSDCell
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
            self.layers.append(
                cell_ctor(
                    d_x=d_h,
                    d_h=d_h,
                    N=ssm_N,
                    shared_mem=shared_mem,
                    **clean_cell_kwargs,
                )
            )

    # ----------------------------------------------------------------------
    def forward(
        self,
        x_seq: torch.Tensor,
        train_mode: bool = True,
        rope_cache: tuple | None = None,
        h0_list: list[torch.Tensor] | None = None,
    ):
        """
        Forward pass through all stacked CRSDCells.
        Returns:
            If train_mode:  (h_seq, h_next_list, k_all, v_all)
            Else:           (h_seq, h_next_list)
        """
        h_seq = self.input_proj(x_seq)
        h_next_list = []
        k_all, v_all = [], []

        if h0_list is None:
            h0_list = [None] * len(self.layers)

        for i, cell in enumerate(self.layers):
            cell.train(mode=train_mode)
            h0 = h0_list[i]

            # -------------------
            # TRAIN MODE
            # -------------------
            if train_mode:
                out = cell(
                    h_seq,
                    h0=h0,
                    train_mode=True,
                    rope_cache=rope_cache,
                )

                # Handle both 2-output (old) and 4-output (new) CRSDCells
                if isinstance(out, (tuple, list)):
                    if len(out) == 4:
                        h_seq, h_state_seq, k, v = out
                        k_all.append(k)
                        v_all.append(v)
                    elif len(out) == 2:
                        h_seq, h_state_seq = out
                    else:
                        raise RuntimeError(
                            f"Unexpected CRSDCell output length {len(out)}. Expected 2 or 4."
                        )
                else:
                    raise RuntimeError("CRSDCell forward must return tuple/list.")


            else:
                h_seq, h_state_seq = cell(
                    h_seq,
                    h0=h0,
                    train_mode=False,
                    rope_cache=rope_cache,
                )

            if h_state_seq is not None:
                h_next_list.append(h_state_seq[:, -1, :, :].detach())
            else:
                h_next_list.append(None)


        if train_mode:
            return h_seq, h_next_list, k_all, v_all
        else:
            return h_seq, h_next_list
