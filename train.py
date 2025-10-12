import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_loader
from models.crsd_seq import CRSDSequence
from utils.config import load_config
from utils.metrics import accuracy_from_logits, bits_per_char, perplexity

def evaluate(model, data_loader, loss_fn):
    """Run evaluation loop and compute metrics."""
    model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits = model(batch)                 # (B, T, V)
            B, T, V = logits.shape
            target = torch.zeros(B, T, dtype=torch.long)

            logits_flat = logits.view(B * T, V)
            target_flat = target.view(B * T)
            loss = loss_fn(logits_flat, target_flat)
            acc = accuracy_from_logits(logits, target)

            total_loss += loss.item()
            total_acc += acc
            count += 1

    avg_loss = total_loss / count
    avg_acc = total_acc / count
    bpc = bits_per_char(avg_loss)
    ppl = perplexity(avg_loss)
    return {"loss": avg_loss, "accuracy": avg_acc, "bpc": bpc, "ppl": ppl}


def train():
    """
    Train CRSD model using configuration defined in experiments/exp_crsd_tiny.yaml.
    """
    # 1️⃣ Load configuration
    config_path = os.path.join("experiments", "exp_crsd_tiny.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ Missing config file: {config_path}")

    cfg = load_config(config_path)
    if cfg is None:
        raise ValueError(f"⚠️ Failed to load or parse config file: {config_path}")

    print(f"✅ Loaded configuration from {config_path}")
    print(cfg)

    # 2️⃣ Data Loader
    data_loader, ds = get_loader(batch=cfg['train']['batch_size'])
    vocab_size = len(ds.tok.inv)
    print(f"📦 Vocab size: {vocab_size}")

    # 3️⃣ Model
    model = CRSDSequence(
        vocab_size=vocab_size,
        emb_dim=cfg['model']['d_x'],
        d_h=cfg['model']['d_h'],
        res_dims=cfg['model']['res_dims'],
        d_k=cfg['model']['d_k'],
        d_v=cfg['model']['d_v'],
        mem_slots=cfg['model']['mem_slots']
    )
    print("🧠 Model initialized successfully.")

    # 4️⃣ Optimizer & Loss
    opt = optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    loss_fn = nn.CrossEntropyLoss()

    # 5️⃣ Training Loop
    for epoch in range(cfg['train']['epochs']):
        model.train()
        total_train_loss = 0.0
        for batch in data_loader:
            batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits = model(batch)                 # (B, T, V)
            B, T, V = logits.shape
            target = torch.zeros(B, T, dtype=torch.long)

            logits_flat = logits.view(B * T, V)
            target_flat = target.view(B * T)
            loss = loss_fn(logits_flat, target_flat)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(data_loader)
        # 🔍 Evaluate each epoch
        metrics = evaluate(model, data_loader, loss_fn)
        print(
            f"🌀 Epoch {epoch+1}/{cfg['train']['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Eval Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.3f} | "
            f"BPC: {metrics['bpc']:.3f} | "
            f"PPL: {metrics['ppl']:.3f}"
        )

    print("✅ Training complete!")


if __name__ == "__main__":
    train()
