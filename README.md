# ResearchLLM

A transformer language model built from scratch in PyTorch, following modern design choices from LLaMA, PaLM, and GPT-4 technical reports. Available in three sizes: 100M, 300M, and 1B parameters.

## Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Normalization | RMSNorm (pre-norm) | Faster, more stable than post-LayerNorm |
| Positional encoding | RoPE | Better length generalization than learned positions |
| Activation | SwiGLU | Higher throughput and expressivity vs GELU |
| Attention | Causal multi-head self-attention | Standard autoregressive setup |
| Tied weights | Yes (embedding ↔ LM head) | ~5–10% parameter saving |
| Biases | None in projections | Modern efficiency pattern |

### Model Sizes

| Variant | Layers | Hidden | Heads | Params |
|---------|--------|--------|-------|--------|
| 100M | 8 | 768 | 12 | ~100M |
| 300M | 16 | 1024 | 16 | ~300M |
| 1B | 14 | 2048 | 16 | ~1B |

## Setup

```bash
pip install torch sentencepiece datasets pyyaml tqdm
```

## Training

### 1. Train the tokenizer (BPE, vocab size 32k)

```bash
python tokenizer/train_tokenizer.py
# Saves tokenizer/researchllm_bpe.model (~hours on full RedPajama)
```

### 2. Run training

```bash
# 100M model (default)
python -m training.train --config configs/researchllm_100m.yaml

# 300M model
python -m training.train --config configs/researchllm_300m.yaml

# 1B model
python -m training.train --config configs/researchllm_1b.yaml

# Start fresh (ignore existing checkpoints)
python -m training.train --config configs/researchllm_100m.yaml --no_resume
```

Training auto-resumes from the latest checkpoint in `checkpoints/`.

Training features:
- Mixed precision (AMP) when CUDA is available
- Cosine LR schedule with linear warmup
- Gradient clipping
- Periodic validation perplexity
- Sample generation logged during training

### Config structure

```yaml
model:
  name: ResearchLLM-100M
  vocab_size: 32000
  max_seq_len: 1024
  num_layers: 8
  hidden_size: 768
  num_heads: 12
  ff_multiplier: 4

training:
  batch_size: 2
  grad_accum_steps: 16
  max_steps: 10000
  warmup_steps: 500
  learning_rate: 3e-4
  weight_decay: 0.1
  grad_clip: 1.0
  checkpoint_interval: 500
  eval_interval: 500
```

## Inference

```python
import torch
from model import ResearchLLM
from inference.generate import generate_text
from training.checkpointing import load_checkpoint

model = ResearchLLM(vocab_size=32000, hidden_size=768, num_layers=8, num_heads=12, max_seq_len=1024)
load_checkpoint("checkpoints/ckpt_0010000.pt", model)
model.eval()

text = generate_text(
    model,
    tokenizer_path="tokenizer/researchllm_bpe.model",
    prompt="The transformer architecture",
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
print(text)
```

## Evaluation

```python
from evaluation.perplexity import compute_perplexity
from training.dataset import StreamingTokenDataset

dataset = StreamingTokenDataset("tokenizer/researchllm_bpe.model", seq_len=1024)
ppl = compute_perplexity(model, dataset, num_batches=100, batch_size=4)
print(f"Perplexity: {ppl:.2f}")
```

## Unit Tests

```bash
python test_model.py      # Full forward pass
python test_attention.py  # Multi-head self-attention
python test_block.py      # Transformer block
python test_mlp.py        # SwiGLU feed-forward
```

## Project Structure

```
research-llm/
├── model/                  # Architecture
│   ├── model.py            # ResearchLLM top-level
│   ├── transformer_block.py
│   ├── attention.py        # Multi-head self-attention + RoPE
│   ├── feedforward.py      # SwiGLU FFN
│   ├── rope.py             # Rotary position embeddings
│   └── rmsnorm.py
├── training/
│   ├── train.py            # Main training loop
│   ├── dataset.py          # Streaming RedPajama dataset
│   ├── optimizer.py        # AdamW with weight decay filtering
│   ├── scheduler.py        # Cosine schedule with warmup
│   └── checkpointing.py    # Save / resume checkpoints
├── inference/
│   ├── generate.py         # Autoregressive text generation
│   └── sampling.py         # Temperature / top-k / top-p
├── evaluation/
│   ├── perplexity.py       # Perplexity on validation set
│   ├── sample_generation.py# Qualitative sample logging
│   └── eval_utils.py       # Shared helpers
├── tokenizer/
│   ├── train_tokenizer.py  # BPE tokenizer training
│   └── test_tokenizer.py
├── configs/
│   ├── researchllm_100m.yaml
│   ├── researchllm_300m.yaml
│   └── researchllm_1b.yaml
└── paper/
    ├── draft.md
    └── references.bib
```

## Data

Training uses [RedPajama-Data-1T-Sample](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) streamed via HuggingFace Datasets. No local storage required.

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{researchllm2026,
  author = {Vignesh Pai},
  title  = {ResearchLLM: A Transformer Language Model Built from Scratch},
  year   = {2026},
  url    = {https://github.com/vigp17/research-llm}
}
```
