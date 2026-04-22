import torch
import math

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, seq_len: int):
    # Handle both 3D (B, T, D) and 4D (B, H, T, D) inputs
    original_shape = x.shape
    if x.ndim == 4:
        B, H, T, D = x.shape
        x = x.reshape(B * H, T, D)
    
    _, _, dim = x.shape
    theta = 10000 ** (-torch.arange(0, dim, 2, device=x.device) / dim)
    positions = torch.arange(seq_len, device=x.device)
    freqs = torch.einsum("i,j->ij", positions, theta)

    sin = freqs.sin()[None, :, :]
    cos = freqs.cos()[None, :, :]

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x = torch.cat([
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos
    ], dim=-1)

    # Restore original shape if it was 4D
    if len(original_shape) == 4:
        x = x.reshape(original_shape)

    return x