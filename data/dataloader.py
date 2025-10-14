import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from .prepare_data import build_tokenizer


class TextDataset(Dataset):
    """
    LLM-style dataset for sequence models (char or subword).
    Stores fixed-length blocks (block_size). Returns (X, Y) where
    X = tokens[:-1], Y = tokens[1:]
    """
    def __init__(self, dataset_path=None, max_len=1024, vocab_size=50000, mode="subword"):
        self.max_len = int(max_len)
        self.tok = build_tokenizer(dataset_path=dataset_path, vocab_size=vocab_size, mode=mode)

        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            from .prepare_data import SENTS
            raw_text = "\n".join(SENTS)

        # Debug: show first few lines of raw text
        print("📝 Sample raw text (first 3 lines):")
        print("\n".join(raw_text.splitlines()[:3]))

        # Encode
        encoded_data = self.tok.encode(raw_text, add_special_tokens=False)
        print(f"🔢 Encoded data length: {len(encoded_data)} tokens")

        # Debug: show first few token IDs and decoded version
        print("🧩 Sample tokens (first 30):", encoded_data[:30])
        try:
            print("🔤 Decoded sample:", self.tok.decode(encoded_data[:50]))
        except Exception as e:
            print("⚠️ Decode error:", e)

        self.block_size = self.max_len
        self.data_blocks = []

        # Determine pad_id safely for various tokenizer implementations
        if hasattr(self.tok, "pad_id"):
            pad_id = self.tok.pad_id
        elif hasattr(self.tok, "tokenizer") and hasattr(self.tok.tokenizer, "pad_token_id"):
            pad_id = self.tok.tokenizer.pad_token_id
        else:
            pad_id = 0

        for i in range(0, len(encoded_data), self.block_size):
            block = encoded_data[i:i + self.block_size]
            if len(block) < self.block_size:
                block += [pad_id] * (self.block_size - len(block))
            self.data_blocks.append(block)

        print(f"📦 Created TextDataset with {len(self.data_blocks)} blocks (each length {self.block_size})")

    def __len__(self):
        return len(self.data_blocks)

    def __getitem__(self, idx):
        token_ids = self.data_blocks[idx]
        X = torch.tensor(token_ids[:-1], dtype=torch.long)
        Y = torch.tensor(token_ids[1:], dtype=torch.long)
        return X, Y


def get_loaders(batch_size=8, max_len=1024, dataset_path=None, num_workers=4, mode="subword", vocab_size=50000, val_split=0.05):
    ds = TextDataset(dataset_path=dataset_path, max_len=max_len, vocab_size=vocab_size, mode=mode)

    # create a small validation split
    total = len(ds)
    n_val = max(1, int(total * float(val_split)))
    n_train = total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    print(f"🧩 Dataset split → Train: {n_train} blocks, Val: {n_val} blocks (total: {total})")

    # DataLoader flags to reduce host<->device stalls
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(0, int(num_workers)),
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **{**loader_kwargs, "shuffle": False})

    print(f"✅ DataLoaders ready | Train batches: {len(train_loader)}, Val batches: {len(val_loader)} | Batch size: {batch_size}")

    # 🔍 Debug: show a few sample batches
    print("\n🔍 Checking sample batches from train_loader:")
    for i, (X, Y) in enumerate(train_loader):
        print(f"Batch {i+1} → X shape: {X.shape}, Y shape: {Y.shape}")
        try:
            sample_decoded = ds.tok.decode(X[0].tolist()[:50])
            print(f"🧠 Decoded X[0][:50]: {sample_decoded}")
        except Exception as e:
            print("⚠️ Decode error in batch sample:", e)
        if i >= 1:
            break  # only print 2 batches

    return train_loader, val_loader, ds.tok
