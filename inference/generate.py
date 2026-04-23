import torch
import sentencepiece as spm

from model import ResearchLLM
from inference.sampling import sample_token


@torch.inference_mode()
def generate(
    model: ResearchLLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_id: int = 2,
) -> torch.Tensor:
    """
    Autoregressively generate tokens from a prompt.

    Args:
        prompt_ids: (1, T) or (T,) tensor of token ids.
        max_new_tokens: maximum tokens to generate.
        eos_id: generation stops when this token is produced.

    Returns:
        (1, T + new_tokens) tensor of all token ids.
    """
    model.eval()
    device = next(model.parameters()).device

    ids = prompt_ids.to(device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)

    max_seq_len = model.token_embedding.num_embeddings  # fallback cap
    # Use the model's own max_seq_len if stored
    cap = getattr(model, "max_seq_len", 2048)

    for _ in range(max_new_tokens):
        # Crop context to max sequence length
        ctx = ids[:, -cap:]
        logits = model(ctx)                  # (1, T, vocab)
        next_logits = logits[:, -1, :]       # (1, vocab)
        next_token = sample_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        ids = torch.cat([ids, next_token], dim=1)
        if next_token.item() == eos_id:
            break

    return ids


def generate_text(
    model: ResearchLLM,
    tokenizer_path: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    prompt_ids = torch.tensor(sp.encode(prompt), dtype=torch.long).unsqueeze(0)
    output_ids = generate(
        model, prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=sp.eos_id(),
    )
    return sp.decode(output_ids[0].tolist())
