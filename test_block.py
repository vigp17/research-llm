import torch
from model.transformer_block import TransformerBlock

block = TransformerBlock(
    hidden_size=768,
    num_heads=12,
    ff_multiplier=4,
)

x = torch.randn(2, 16, 768)
y = block(x)

print(y.shape)