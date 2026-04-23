import torch
import torch.nn.functional as F


def compute_batch_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


def get_device(model) -> torch.device:
    return next(model.parameters()).device


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()
