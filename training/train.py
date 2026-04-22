import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.model import ResearchLLM
from training.dataset import StreamingTokenDataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResearchLLM(
        vocab_size=32000,
        hidden_size=768,
        num_layers=8,        # ResearchLLM‑100M
        num_heads=12,
        ff_multiplier=4,
        max_seq_len=1024,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
    )

    dataset = StreamingTokenDataset(
        tokenizer_path="tokenizer/researchllm_bpe.model",
        seq_len=1024,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,    # we’ll accumulate gradients
    )

    grad_accum_steps = 16
    model.train()

    for step, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            print(f"step {step} | loss {loss.item() * grad_accum_steps:.4f}")

        if step >= 500:
            break


if __name__ == "__main__":
    train()