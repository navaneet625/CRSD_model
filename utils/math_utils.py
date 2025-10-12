import torch

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))

def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return e_x / e_x.sum(dim=dim, keepdim=True)
