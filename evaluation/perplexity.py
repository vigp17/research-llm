import math
import torch
from torch.utils.data import DataLoader

from evaluation.eval_utils import compute_batch_loss, get_device


def compute_perplexity(model, dataset, num_batches: int = 100, batch_size: int = 4) -> float:
    """
    Estimate perplexity on a dataset (lower is better).

    Returns perplexity = exp(average cross-entropy loss).
    """
    model.eval()
    device = get_device(model)

    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    count = 0

    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        loss = compute_batch_loss(model, x, y)
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    return math.exp(avg_loss)
