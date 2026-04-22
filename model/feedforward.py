import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ff_multiplier: int = 4):
        super().__init__()

        inner_dim = hidden_size * ff_multiplier

        self.w1 = nn.Linear(hidden_size, inner_dim, bias=False)
        self.w2 = nn.Linear(hidden_size, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))