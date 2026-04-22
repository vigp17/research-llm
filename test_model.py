import torch
from model.model import ResearchLLM

model = ResearchLLM(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    ff_multiplier=4,
    max_seq_len=1024,
)

input_ids = torch.randint(0, 32000, (2, 16))
logits = model(input_ids)

print(logits.shape)
