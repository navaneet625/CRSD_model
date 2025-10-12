import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_loader
from data.prepare_data import build_tokenizer
from models.crsd_seq import CRSDSequence

def train(config_path):
    cfg = yaml.safe_load(open(config_path))
    data_loader, ds = get_loader(batch=cfg['train']['batch_size'])
    vocab_size = len(ds.tok.inv)
    model = CRSDSequence(vocab_size=vocab_size, emb_dim=cfg['model']['d_x'],
                         d_h=cfg['model']['d_h'], res_dims=cfg['model']['res_dims'],
                         d_k=cfg['model']['d_k'], d_v=cfg['model']['d_v'], mem_slots=cfg['model']['mem_slots'])
    opt = optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg['train']['epochs']):
        for batch in data_loader:
            # batch: (B, L) but our TinyDataset returns variable lengths — handle single sample
            batch = batch[0]
            logits = model(batch)
            # naive target: shift left (language modeling) — for demo use zeros
            target = torch.zeros(logits.shape[0], dtype=torch.long)
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch} loss {loss.item():.4f}")

if __name__ == '__main__':
    train('experiments/exp_crsd_tiny.yaml')