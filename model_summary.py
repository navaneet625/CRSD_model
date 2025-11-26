import os
import torch
import torch.nn as nn
from torchinfo import summary
from utils.config import load_config
from models.crsd_seq import CRSDSequence


def print_crsd_summary(
    model: nn.Module,
    vocab_size: int,
    seq_len: int = 64,
    batch_size: int = 8,
    device: str = "cpu",
):
    """
    Prints a detailed layer summary for CRSDSequence, like Mamba-style model summary.

    Args:
        model: CRSDSequence instance
        vocab_size: Vocabulary size (for embedding)
        seq_len: Sequence length for dummy input
        batch_size: Batch size for dummy input
        device: 'cpu' or 'cuda'
    """
    # Move model to selected device
    model = model.to(device)

    # Use consistent float32 for summary to avoid dtype mismatches
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    if hasattr(model, "kcm"):
        # Force KCM buffers to device and dtype for summary
        model.kcm.M = model.kcm.M.to(device=device, dtype=torch.float32)
        model.kcm.dtype = torch.float32
        model.kcm.phi_proj = model.kcm.phi_proj.to(dtype=torch.float32)

    print("=" * 45)
    print("🧠 Model Summary — CRSD Memory-Augmented Sequence Model")
    print("=" * 45)
    print(f"Vocab Size : {vocab_size}")
    print(f"Input Shape: ({batch_size}, {seq_len})")
    print(f"Device     : {device}")
    print("---------------------------------------------")

    # --- Safe summary wrapper (torchinfo can crash on large recursive models) ---
    try:
        summary(
            model,
            input_data=x,
            depth=4,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ),
            row_settings=["var_names"],
            verbose=1,
        )
    except Exception as e:
        print("⚠️ torchinfo failed during forward pass:")
        print(e)
        print("Continuing with parameter count only.\n")

    # --- Manual parameter statistics ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable

    print("=" * 90)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {frozen:,}")
    print("=" * 90)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "experiments", "exp_language_model.yaml")

    cfg = load_config(config_path)
    mcfg = cfg["model"]

    # ---- 🔧 Sanitize dtype ----
    if isinstance(mcfg.get("memory_dtype"), str):
        mcfg["memory_dtype"] = getattr(torch, mcfg["memory_dtype"], torch.float32)

    # ---- 🔧 Ensure numeric types ----
    for key in ["emb_dim", "d_x", "d_h", "d_k", "d_v", "rank_kcm", "layers"]:
        if key in mcfg:
            mcfg[key] = int(mcfg[key])

    vocab_size = 259

    # ---- Instantiate CRSDSequence ----
    model = CRSDSequence(
        vocab_size=vocab_size,
        emb_dim=mcfg["emb_dim"],
        d_x=mcfg["d_x"],
        **{k: v for k, v in mcfg.items() if k not in ["emb_dim", "d_x"]},
    )

    # ---- Print summary ----
    print_crsd_summary(model, vocab_size=vocab_size, seq_len=64, batch_size=8)
