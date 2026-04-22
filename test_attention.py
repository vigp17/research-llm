import torch
from model.attention import MultiHeadSelfAttention

attn = MultiHeadSelfAttention(hidden_size=768, num_heads=12)
x = torch.randn(2, 16, 768)

y = attn(x)
print(y.shape)
