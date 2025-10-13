import os
import torch
from torch.utils.data import Dataset, DataLoader
from .prepare_data import build_tokenizer

class TextDataset(Dataset):
    """
    LLM-style dataset for sequence models (char or subword).
    """
    def __init__(self, dataset_path=None, max_len=1024, vocab_size=50000, mode="subword"):
        self.max_len = max_len
        self.tok = build_tokenizer(dataset_path=dataset_path, vocab_size=vocab_size, mode=mode)
        
        # Load data
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            from .prepare_data import SENTS
            raw_text = "\n".join(SENTS)

        encoded_data = self.tok.encode(raw_text, add_special_tokens=False)
        self.block_size = max_len
        self.data_blocks = []

        # ✅ Determine pad_id safely for all tokenizer types
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

    def __len__(self):
        return len(self.data_blocks)

    def __getitem__(self, idx):
        token_ids = self.data_blocks[idx]
        X = torch.tensor(token_ids[:-1], dtype=torch.long)
        Y = torch.tensor(token_ids[1:], dtype=torch.long)
        return X, Y


def get_loader(batch_size=8, max_len=1024, dataset_path=None, num_workers=4, mode="subword"):
    ds = TextDataset(dataset_path=dataset_path, max_len=max_len, mode=mode)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    return loader, ds
