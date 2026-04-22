import torch
from model.feedforward import SwiGLUFeedForward

mlp = SwiGLUFeedForward(hidden_size=768)
x = torch.randn(2, 16, 768)

y = mlp(x)
print(y.shape)