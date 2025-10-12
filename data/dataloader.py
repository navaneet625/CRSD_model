# data/dataloader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from .prepare_data import build_tokenizer


class TextDataset(Dataset):
    """
    Character- or word-level dataset for CRSD models.
    Supports both small demo text and large corpus files.
    """

    def __init__(self, dataset_path=None, max_len=128):
        self.dataset_path = dataset_path
        self.max_len = max_len
        self.tok = build_tokenizer(dataset_path)

        # Load data
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            # split into sentences or chunks
            self.samples = raw_text.split("\n")
        else:
            from .prepare_data import SENTS
            self.samples = SENTS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx].strip()
        ids = self.tok.encode(s)

        # Pad/truncate to fixed length
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))  # <PAD>=0
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)


def get_loader(batch=2, max_len=128, dataset_path=None):
    """
    Return DataLoader + dataset object.
    Works for both toy and large datasets.
    """
    ds = TextDataset(dataset_path=dataset_path, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    return loader, ds
