# tiny dataloader for demonstration
import torch
from .prepare_data import build_tokenizer

class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tok = build_tokenizer()
        self.sents = [s for s in __import__('data.prepare_data', fromlist=['SENTS']).SENTS]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        s = self.sents[idx]
        ids = self.tok.encode(s)
        return torch.tensor(ids, dtype=torch.long)

def get_loader(batch=1):
    ds = TinyDataset()
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True), ds