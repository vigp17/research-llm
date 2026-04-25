import argparse
import os
import shutil
import time

import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ResearchLLM
from training.checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
from training.dataset import StreamingTokenDataset
from training.optimizer import build_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from evaluation.perplexity import compute_perplexity
from evaluation.sample_generation import log_samples


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> ResearchLLM:
    m = cfg["model"]
    model = ResearchLLM(
        vocab_size=m["vocab_size"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        ff_multiplier=m.get("ff_multiplier", 4),
        max_seq_len=m["max_seq_len"],
    ).to(device)
    model.max_seq_len = m["max_seq_len"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {m['name']}  |  params: {n_params / 1e6:.1f}M")
    return model


def train(config_path: str, tokenizer_path: str, checkpoint_dir: str, resume: bool = True):
    cfg = load_config(config_path)
    t = cfg.get("training", {})

    # ── Hyperparameters ──────────────────────────────────────────────────────
    batch_size       = t.get("batch_size", 2)
    grad_accum       = t.get("grad_accum_steps", 16)
    max_steps        = t.get("max_steps", 10000)
    warmup_steps     = t.get("warmup_steps", 500)
    lr               = float(t.get("learning_rate", 3e-4))
    weight_decay     = float(t.get("weight_decay", 0.1))
    grad_clip        = float(t.get("grad_clip", 1.0))
    ckpt_interval    = t.get("checkpoint_interval", 500)
    eval_interval    = t.get("eval_interval", 500)
    seq_len          = cfg["model"]["max_seq_len"]
    use_amp          = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model / optimiser / scheduler ───────────────────────────────────────
    model     = build_model(cfg, device)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    scaler    = GradScaler(enabled=use_amp)

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataset = StreamingTokenDataset(tokenizer_path, seq_len=seq_len, split="train")
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=use_amp)

    # Optional small validation split (same dataset shuffled differently)
    val_dataset = StreamingTokenDataset(tokenizer_path, seq_len=seq_len, split="train")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt_path = latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            start_step, last_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler, device=str(device))
            print(f"Resumed from {ckpt_path}  (step {start_step}, loss {last_loss:.4f})")

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    step         = start_step
    accum_loss   = 0.0
    t0           = time.time()

    data_iter = iter(train_loader)

    pbar = tqdm(total=max_steps, initial=start_step, desc="Training", dynamic_ncols=True)

    while step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss   = loss / grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            cur_loss = accum_loss
            cur_lr   = scheduler.get_last_lr()[0]
            elapsed  = time.time() - t0

            pbar.set_postfix(loss=f"{cur_loss:.4f}", lr=f"{cur_lr:.2e}", s=f"{elapsed:.0f}s")
            pbar.update(grad_accum)

            accum_loss = 0.0
            t0 = time.time()

        step += 1

        # ── Checkpoint ───────────────────────────────────────────────────────
        if step % ckpt_interval == 0:
            ckpt_file = os.path.join(checkpoint_dir, f"ckpt_{step:07d}.pt")
            tmp_local = f"/tmp/ckpt_{step:07d}.pt"
            save_checkpoint(tmp_local, model, optimizer, scheduler, step, cur_loss)
            shutil.move(tmp_local, ckpt_file)
            tqdm.write(f"[step {step}] checkpoint saved → {ckpt_file}")

        # ── Evaluation ───────────────────────────────────────────────────────
        if step % eval_interval == 0:
            model.eval()
            ppl = compute_perplexity(model, val_dataset, num_batches=50, batch_size=batch_size)
            tqdm.write(f"[step {step}] val perplexity: {ppl:.2f}")
            log_samples(model, tokenizer_path, step, max_new_tokens=80)
            model.train()

    pbar.close()

    # Final checkpoint
    final_ckpt = os.path.join(checkpoint_dir, f"ckpt_{step:07d}.pt")
    tmp_local = f"/tmp/ckpt_{step:07d}.pt"
    save_checkpoint(tmp_local, model, optimizer, scheduler, step, cur_loss)
    os.replace(tmp_local, final_ckpt)
    print(f"Training complete. Final checkpoint → {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResearchLLM")
    parser.add_argument("--config",          default="configs/researchllm_100m.yaml")
    parser.add_argument("--tokenizer",       default="tokenizer/researchllm_bpe.model")
    parser.add_argument("--checkpoint_dir",  default="checkpoints")
    parser.add_argument("--no_resume",       action="store_true")
    args = parser.parse_args()

    train(
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
    )
