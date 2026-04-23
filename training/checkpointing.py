import os
import torch


def save_checkpoint(path: str, model, optimizer, scheduler, step: int, loss: float, keep_last: int = 2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save({
        "step": step,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }, tmp_path)
    os.replace(tmp_path, path)

    # Delete old checkpoints, keeping only the last `keep_last`
    ckpt_dir = os.path.dirname(path)
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda f: int(f.split("_")[-1].replace(".pt", "")),
    )
    for old in ckpts[:-keep_last]:
        os.remove(os.path.join(ckpt_dir, old))


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt.get("loss", float("inf"))


def latest_checkpoint(checkpoint_dir: str):
    """Return path to the most recent valid checkpoint, or None if none exist."""
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
        key=lambda f: int(f.split("_")[-1].replace(".pt", "")),
        reverse=True,
    )
    for ckpt in ckpts:
        path = os.path.join(checkpoint_dir, ckpt)
        try:
            torch.load(path, map_location="cpu", weights_only=True)
            return path
        except Exception:
            continue
    return None
