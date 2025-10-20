import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import pickle

from data.dataloader import get_loaders
from models.crsd_seq import CRSDSequence
from utils.config import load_config
from utils.metrics import accuracy_from_logits, bits_per_char, perplexity

torch._dynamo.config.capture_scalar_outputs = True

# ---------------- EVAL ----------------
@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device, use_amp=False):
    model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(X, train_mode=False)
            B, Tm1, V = logits.shape
            loss = loss_fn(logits.view(B * Tm1, V), Y.view(B * Tm1))
        acc = accuracy_from_logits(logits, Y)
        total_loss += loss.item()
        total_acc += acc
        count += 1

    avg_loss = total_loss / max(1, count)
    avg_acc = total_acc / max(1, count)
    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "bpc": bits_per_char(avg_loss),
        "ppl": perplexity(avg_loss),
    }


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ---------------- TRAIN ----------------
def train():
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "experiments", "exp_language_model.yaml")

    if not os.path.exists(config_path):
        print("⚠️ Could not find config file. Exiting.")
        sys.exit(1)

    # Load config
    try:
        cfg = load_config(config_path)
    except Exception:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"💻 Using device: {device}")
    use_amp = device.type == "cuda"

    # Data
    dataset_path = os.path.join(project_root, "data", data_cfg.get("dataset_path", "text.txt"))
    train_loader, val_loader, tok = get_loaders(
        batch_size=train_cfg.get("batch_size", 64),
        max_len=data_cfg.get("max_len", 256),
        dataset_path=dataset_path,
        num_workers=data_cfg.get("num_workers", 2),
        mode=data_cfg.get("data_mode", "char"),
        vocab_size=data_cfg.get("vocab_size", 50000),
    )
    vocab_size = tok.vocab_size() if hasattr(tok, "vocab_size") else 50000
    print(f"✅ Loaded data — vocab size: {vocab_size}")

    # ✅ ADDED: Save tokenizer safely once
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.pt")
    torch.save({
        "tokenizer": tok,
        "vocab_size": vocab_size,
        "data_mode": data_cfg.get("data_mode", "char"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S")
    }, tokenizer_path)
    print(f"✅ Tokenizer saved at: {tokenizer_path}")

    # Model
    model = CRSDSequence(
        vocab_size=vocab_size,
        emb_dim=model_cfg["emb_dim"],
        d_x=model_cfg["d_x"],
        d_h=model_cfg["d_h"],
        ssm_N=model_cfg["ssm_N"],
        d_k=model_cfg["d_k"],
        d_v=model_cfg["d_v"],
        rank_kcm=model_cfg.get("rank_kcm", 64),
        use_rope=model_cfg.get("use_rope", True),
        auto_write=model_cfg.get("auto_write", True),
        res_dropout_p=model_cfg.get("res_dropout_p", 0.0),
        use_diag_A=model_cfg.get("use_diag_A", True),
        memory_dtype=torch.float32,
        layers=model_cfg.get("layers", 2),
        kcm_consolidation_rate=model_cfg.get("kcm_consolidation_rate", 0.1),
        kcm_reduce=model_cfg.get("kcm_reduce", "sum"),
        max_len=data_cfg.get("max_len", 256),
    ).to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"🧩 Using DataParallel over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Optimizer & Loss
    opt = optim.Adam(model.parameters(),
                     lr=train_cfg.get("lr", 3e-4),
                     betas=train_cfg.get("betas", [0.9, 0.98]))
    loss_fn = nn.CrossEntropyLoss()

    try:
        scaler = torch.amp.GradScaler(device_type=device.type, enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    grad_clip = train_cfg.get("grad_clip", 1.0)
    epochs = train_cfg.get("epochs", 10)
    ckpt_every = train_cfg.get("checkpoint_every", 1)

    print("🚀 Starting training...\n")

    # ---------------- MAIN LOOP ----------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        avg_loss = float('nan')

        print(f"\n🌙 Epoch {epoch + 1}/{epochs}")
        print("-" * 60)

        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(X, train_mode=True)
                B, Tm1, V = logits.shape
                loss = loss_fn(logits.view(B * Tm1, V), Y.view(B * Tm1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress = (batch_idx + 1) / len(train_loader) * 100
            sys.stdout.write(
                f"\r🧩 Step [{batch_idx + 1}/{len(train_loader)}] "
                f"({progress:6.2f}%) | Loss: {loss.item():.4f} | "
                f"Avg: {avg_loss:.4f}"
            )
            sys.stdout.flush()

        print()
        metrics = evaluate(model, val_loader, loss_fn, device, use_amp)
        eval_loss = metrics["loss"]
        print(
            f"🌀 Epoch {epoch + 1}/{epochs} | Train: {avg_loss:.4f} | Eval: {eval_loss:.4f} "
            f"| Acc: {metrics['accuracy']:.3f} | BPC: {metrics['bpc']:.3f} | PPL: {metrics['ppl']:.3f}"
        )

        # ✅ ADDED: Save checkpoint with memory_M + config + tokenizer info
        if (epoch + 1) % ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"crsd_epoch{epoch + 1:02d}.pt")
            state = {
                "epoch": epoch + 1,
                "model_state": _unwrap(model).state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "vocab_size": vocab_size,
                "config": cfg,
            }

            # ✅ Save memory M from KCM if available
            m = _unwrap(model)
            if hasattr(m, "kcm") and hasattr(m.kcm, "M"):
                state["memory_M"] = m.kcm.M.detach().cpu()

            torch.save(state, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    print("\n✅ Training complete! 🚀")


if __name__ == "__main__":
    train()
