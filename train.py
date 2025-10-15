import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import importlib 

# local imports
from data.dataloader import get_loaders
from models.crsd_seq import CRSDSequence
from utils.config import load_config
from utils.metrics import accuracy_from_logits, bits_per_char, perplexity
import torch
torch._dynamo.config.capture_scalar_outputs = True

@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device, use_amp=False):
    model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0

    for batch_data in data_loader:
        X, Y = batch_data
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        # use recommended API: torch.amp.autocast with device_type
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(X, train_mode=False)
            B, T_minus_1, V = logits.shape
            logits_flat = logits.view(B * T_minus_1, V)
            target_flat = Y.view(B * T_minus_1)
            loss = loss_fn(logits_flat, target_flat)

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


def validate_config(cfg):
    m = cfg["model"]
    assert m["d_x"] > 0 and m["d_h"] > 0, "d_x and d_h must be > 0"
    assert sum(m["res_dims"]) <= m["d_h"], "sum(res_dims) should not exceed d_h"
    assert m["d_k"] == m["d_h"], "d_k should match d_h for best stability"
    print("*"*60)
    print(f"[Config OK] R={sum(m['res_dims'])}, d_h={m['d_h']}, d_k={m['d_k']}, d_v={m['d_v']}")
    print("*"*60)


def train():
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "experiments", "exp_language_model.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ Missing config file: {config_path}")

    try:
        cfg = load_config(config_path)
    except Exception:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    print(f"✅ Loaded configuration from {config_path}")

    mode = cfg.get("mode", "small").lower()
    if f"{mode}_model" in cfg:
        model_cfg = cfg[f"{mode}_model"]
        train_cfg = cfg.get(f"{mode}_train", {})
        data_cfg = cfg.get(f"{mode}_data", {})
    else:
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("train", {})
        data_cfg = cfg.get("data", {})

    print(f"⚙️  Using {mode.upper()} configuration")

    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # AMP only supported on CUDA here (safe default)
    use_amp = (device.type == "cuda")

    # speed flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    data_mode = data_cfg.get("data_mode", "subword").lower()
    print(f"🔡 Tokenizer mode: {data_mode}")

    train_loader, val_loader, tok = get_loaders(
        batch_size=train_cfg.get("batch_size", 64),
        max_len=data_cfg.get("max_len", 256),
        dataset_path=data_cfg.get("dataset_path", None),
        num_workers=data_cfg.get("num_workers", 2),
        mode=data_cfg.get("data_mode", "char"),
        vocab_size=data_cfg.get("vocab_size", 50000),
    )

    # safe vocab size extraction
    try:
        vocab_size = int(tok.vocab_size())
    except Exception:
        try:
            vocab_size = int(getattr(tok, "vocab_size", None) or len(getattr(tok, "vocab", [])))
        except Exception:
            vocab_size = 50000


    validate_config(cfg)

    model = CRSDSequence(
        vocab_size=vocab_size,
        emb_dim=model_cfg["emb_dim"],
        d_x=model_cfg["d_x"],
        d_h=model_cfg["d_h"],
        res_dims=model_cfg["res_dims"],
        d_k=model_cfg["d_k"],
        d_v=model_cfg["d_v"],
        mem_slots=model_cfg["mem_slots"],
        hebbian_topk=model_cfg.get("hebbian_topk", 16),
        episodic_topk=model_cfg.get("episodic_topk", 8),
        auto_write=model_cfg.get("auto_write", True),
    ).to(device)

    # -------------------------
    # Conditionally compile model
    # -------------------------
    do_compile = hasattr(torch, "compile")
    compile_ok = False

    if do_compile and device.type == "cuda":
        try:
            # detect compute capability
            dev_idx = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(dev_idx)
            print(f"🔎 CUDA compute capability detected: {major}.{minor}")
            if major >= 7:
                compile_ok = True
            else:
                print("⚠️ GPU compute capability < 7.0: skipping torch.compile() (Triton requires >=7.0).")
        except Exception as e:
            print("⚠️ Couldn't determine device capability, skipping compile. Error:", e)
            compile_ok = False
    else:
        if do_compile:
            print("ℹ️ torch.compile available but not on CUDA device — skipping compilation.")
        else:
            print("ℹ️ torch.compile not available in this PyTorch build.")

    if do_compile and compile_ok:
        try:
            # avoid rebinding 'torch' by importing torch._dynamo via importlib
            try:
                torch_dynamo = importlib.import_module("torch._dynamo")
                torch_dynamo.config.suppress_errors = True
            except Exception:
                # if torch._dynamo missing, just continue
                pass
        except Exception:
            pass

        try:
            model = torch.compile(model, backend="inductor")
            print("🔧 Model compiled with torch.compile()")
        except Exception as e:
            print("⚠️ torch.compile() failed or raised; continuing with eager execution. Error:", e)
    else:
        print("🔧 Running model in eager mode (no torch.compile()).")

    print("🧩 Model initialized successfully.")

    opt = optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3))
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    grad_clip = train_cfg.get("grad_clip", 1.0)
    epochs = train_cfg.get("epochs", 5)
    ckpt_every = train_cfg.get("checkpoint_every", 1)

    os.makedirs("checkpoints", exist_ok=True)
    best_eval_loss = float("inf")

    print("🚀 Starting training...\n")
    total_batches = len(train_loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        print(f"\n🌙 Epoch {epoch + 1}/{epochs}")
        print("-" * 60)

        for batch_idx, batch_data in enumerate(train_loader):
            X, Y = batch_data
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            # use recommended API: torch.amp.autocast with device_type
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(X, train_mode=True)
                B, T_minus_1, V = logits.shape
                loss = loss_fn(logits.view(B * T_minus_1, V), Y.view(B * T_minus_1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress = (batch_idx + 1) / total_batches * 100
            eta = (time.time() - start_time) / (batch_idx + 1) * (total_batches - batch_idx - 1)

            sys.stdout.write(
                f"\r🧩 Step [{batch_idx + 1}/{total_batches}] "
                f"({progress:6.2f}%) | Loss: {loss.item():.4f} | "
                f"Avg: {avg_loss:.4f} | ETA: {eta/60:.1f}m"
            )
            sys.stdout.flush()

        print()
        avg_train_loss = total_loss / max(1, total_batches)
        elapsed = time.time() - start_time

        metrics = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp)
        eval_loss = metrics["loss"]
        print(
            f"🌀 Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.3f} | "
            f"BPC: {metrics['bpc']:.3f} | "
            f"PPL: {metrics['ppl']:.3f} | "
            f"⏱️ {elapsed:.1f}s"
        )

        if (epoch + 1) % ckpt_every == 0:
            ckpt_path = f"checkpoints/crsd_epoch{epoch + 1}.pt"
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_path = "checkpoints/crsd_best.pt"
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1
            }, best_path)
            print(f"🏅 New best model saved: {best_path}")

    print("\n✅ Training complete! 🚀")


if __name__ == "__main__":
    train()
