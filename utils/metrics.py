import torch
import math

# -----------------------------
# 1️⃣ Standard Sequence Metrics
# -----------------------------
def accuracy_from_logits(logits, targets):
    """
    logits: (B, T, V)
    targets: (B, T)
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()


def bits_per_char(loss):
    """
    Convert CrossEntropy loss from nats to bits
    """
    return loss / math.log(2)


def perplexity(loss):
    """
    Standard perplexity (for language models)
    """
    return math.exp(loss)


def mse(pred, target):
    """
    Mean Squared Error
    """
    return torch.mean((pred - target) ** 2).item()


# -----------------------------
# 2️⃣ Memory-Specific Metrics
# -----------------------------
def episodic_recall_accuracy(recalled, target, threshold=1e-3):
    """
    recalled: (B, d_v)  --> output from episodic memory
    target: (B, d_v)    --> expected memory value
    threshold: float, similarity cutoff
    Returns fraction of recalled vectors close enough to target
    """
    sim = F.cosine_similarity(recalled, target, dim=-1)  # (B,)
    correct = (sim > (1 - threshold)).float()
    return correct.mean().item()


def hebbian_association_score(recalled, target):
    """
    Cosine similarity between Hebbian memory recall and target
    """
    sim = F.cosine_similarity(recalled, target, dim=-1)
    return sim.mean().item()


def memory_utilization(alpha):
    """
    alpha: (B, slots) attention weights from memory recall
    Measures fraction of slots actively used (alpha > 1e-3)
    """
    active = (alpha > 1e-3).float()
    return active.mean().item()


def memory_entropy(alpha, eps=1e-8):
    """
    alpha: (B, slots) softmax attention weights
    Returns entropy of memory recall distribution
    """
    H = - (alpha * torch.log(alpha + eps)).sum(dim=-1)  # (B,)
    return H.mean().item()


def memory_retention_score(recalled, original):
    """
    Measures how well memory retains a specific value over time
    """
    sim = F.cosine_similarity(recalled, original, dim=-1)
    return sim.mean().item()


# -----------------------------
# 3️⃣ Combined Evaluation Summary
# -----------------------------
def evaluate_model(logits=None, targets=None,
                   pred=None, target=None,
                   mem_recall=None, mem_target=None,
                   mem_alpha=None):
    """
    Returns a dict of all relevant metrics
    """
    metrics = {}

    if logits is not None and targets is not None:
        metrics['accuracy'] = accuracy_from_logits(logits, targets)
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  targets.view(-1))
        metrics['bpc'] = bits_per_char(ce_loss.item())
        metrics['perplexity'] = perplexity(ce_loss.item())

    if pred is not None and target is not None:
        metrics['mse'] = mse(pred, target)

    if mem_recall is not None and mem_target is not None:
        metrics['episodic_recall_acc'] = episodic_recall_accuracy(mem_recall, mem_target)
        metrics['hebbian_assoc_score'] = hebbian_association_score(mem_recall, mem_target)

    if mem_alpha is not None:
        metrics['memory_utilization'] = memory_utilization(mem_alpha)
        metrics['memory_entropy'] = memory_entropy(mem_alpha)

    return metrics