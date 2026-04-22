import torch
from model.model import ResearchLLM

model = ResearchLLM(
    vocab_size=32000,
    hidden_size=768,
    num_layers=8,
    num_heads=12,
    ff_multiplier=4,
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params / 1e6:.2f}M")
print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")