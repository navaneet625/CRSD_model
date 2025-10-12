import torch
import math

def accuracy_from_logits(logits, targets):
    # logits: (B, T, V)
    # targets: (B, T)
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()

def bits_per_char(loss):
    # CrossEntropy loss is in nats; convert to bits
    return loss / math.log(2)

def perplexity(loss):
    # Only for language models
    return math.exp(loss)

def mse(pred, target):
    return torch.mean((pred - target) ** 2).item()
