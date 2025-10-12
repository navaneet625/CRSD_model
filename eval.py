import torch
import torch.nn.functional as F
from utils.memory_utils import normalize
from utils.eval_utils import bits_per_char, perplexity, accuracy_from_logits

def evaluate(model, dataloader, task_type="lm"):
    metrics = {}
    model.eval()
    with torch.no_grad():
        losses, accs = [], []
        for batch in dataloader:
            logits = model(batch)
            B, T, V = logits.shape
            target = torch.zeros(B, T, dtype=torch.long)
            loss = F.cross_entropy(logits.view(B*T, V), target.view(B*T))
            losses.append(loss.item())
            if task_type in ["lm", "classification"]:
                accs.append(accuracy_from_logits(logits, target))

    avg_loss = sum(losses) / len(losses)
    metrics["loss"] = avg_loss
    metrics["bpc"] = bits_per_char(avg_loss)
    metrics["ppl"] = perplexity(avg_loss)
    if accs:
        metrics["accuracy"] = sum(accs) / len(accs)
    return metrics
