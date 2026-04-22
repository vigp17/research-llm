import torch
import torch.nn as nn

from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm


class ResearchLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ff_multiplier: int = 4,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_multiplier=ff_multiplier,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(hidden_size)

        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights (common modern practice)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T)
        returns logits: (B, T, vocab_size)
        """
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
