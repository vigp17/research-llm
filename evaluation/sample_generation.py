from inference.generate import generate_text
from model import ResearchLLM

DEFAULT_PROMPTS = [
    "The transformer architecture",
    "Once upon a time",
    "In the field of machine learning",
    "The scientific method",
]


def log_samples(
    model: ResearchLLM,
    tokenizer_path: str,
    step: int,
    prompts: list[str] = DEFAULT_PROMPTS,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
):
    print(f"\n{'='*60}")
    print(f"Samples at step {step}")
    print(f"{'='*60}")
    for prompt in prompts:
        text = generate_text(
            model, tokenizer_path, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        print(f"\nPrompt: {prompt!r}")
        print(f"Output: {text!r}")
    print(f"{'='*60}\n")
