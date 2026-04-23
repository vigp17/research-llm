import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.95)):
    """AdamW with weight decay only on weight matrices, not biases or norms."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())
