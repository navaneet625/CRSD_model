import torch

def normalize(x, eps=1e-6):
    return x / (x.norm() + eps)

