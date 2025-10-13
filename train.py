import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_loader
from models.crsd_seq import CRSDSequence
from utils.config import load_config
from utils.metrics import accuracy_from_logits, bits_per_char, perplexity
import yaml
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false" 


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    
    for batch_data in data_loader:
        X, Y = batch_data 
        X = X.to(device)
        Y = Y.to(device)

        logits = model(X)

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

    data_mode = data_cfg.get("data_mode", "subword").lower()  # NEW 🧠
    print(f"🔡 Tokenizer mode: {data_mode}")

    data_loader, ds = get_loader(
        batch_size=train_cfg.get("batch_size", 1), 
        max_len=data_cfg.get("max_len", 128),
        dataset_path=data_cfg.get("dataset_path", None),
        mode=data_mode,
    )
    vocab_size = ds.tok.vocab_size() 

    model = CRSDSequence(
        vocab_size=vocab_size,
        emb_dim=model_cfg["d_x"],
        **model_cfg,
    ).to(device)
    print("🧩 Model initialized successfully.")

    opt = optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3))
    loss_fn = nn.CrossEntropyLoss()

    grad_clip = train_cfg.get("grad_clip", 1.0)
    epochs = train_cfg.get("epochs", 5)
    ckpt_every = train_cfg.get("checkpoint_every", 1)

    os.makedirs("checkpoints", exist_ok=True)
    best_eval_loss = float("inf")

    print("🚀 Starting training...\n")
    total_batches = len(data_loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        print(f"\n🌙 Epoch {epoch + 1}/{epochs}")
        print("-" * 60)

        for batch_idx, batch_data in enumerate(data_loader):
            X, Y = batch_data 
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)

            B, T_minus_1, V = logits.shape
            
            # Flatten Logits and Targets for CrossEntropyLoss
            loss = loss_fn(logits.view(B * T_minus_1, V), Y.view(B * T_minus_1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

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

        # Epoch summary
        avg_train_loss = total_loss / total_batches
        elapsed = time.time() - start_time

        metrics = evaluate(model, data_loader, loss_fn, device) 
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
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        # 🔥 Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_path = "checkpoints/crsd_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"🏅 New best model saved: {best_path}")

    print("\n✅ Training complete! 🚀")


if __name__ == "__main__":
    train()