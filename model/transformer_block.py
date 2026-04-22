import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .feedforward import SwiGLUFeedForward
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ff_multiplier: int = 4,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUFeedForward(
            hidden_size=hidden_size,
            ff_multiplier=ff_multiplier,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block (pre-norm)
        x = x + self.attn(self.attn_norm(x))

        # Feedforward block (pre-norm)
        x = x + self.ffn(self.ffn_norm(x))

        return x