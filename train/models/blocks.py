from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from .ops.attention import MultiheadAttentionFlash

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor):
        x2 = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(x2 + self.eps)
        return self.weight * x

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)

class TinyDiTBlock(nn.Module):
    """Minimal DiT-style block for sanity checks."""
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0, causal: bool = False, use_flash_attn: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = MultiheadAttentionFlash(dim, n_heads, dropout=dropout, causal=causal, use_flash_attn=use_flash_attn)
        self.norm2 = RMSNorm(dim)
        self.mlp   = MLP(dim, hidden_mult=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TinyDiT(nn.Module):
    def __init__(self, dim: int, n_heads: int, depth: int, mlp_ratio: int = 4, dropout: float = 0.0, causal: bool = False, use_flash_attn: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            TinyDiTBlock(dim, n_heads, mlp_ratio, dropout, causal, use_flash_attn) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor):
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)