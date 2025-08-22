# TinyDiT: minimal implementation of a DiT-style transformer for testing
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from .ops.attention import MultiheadAttentionFlash

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # learnable scaling parameter
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # RMS along the last dimension and normalize
        x2 = x.pow(2).mean(-1, keepdim=True)  # [..., 1]
        x = x * torch.rsqrt(x2 + self.eps)
        
        # learnable scaling
        return self.weight * x

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * hidden_mult
        
        # Sequential network: Linear -> GELU -> Dropout -> Linear
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim))
    
    def forward(self, x: torch.Tensor):
        return self.net(x)

class TinyDiTBlock(nn.Module):
    """
    Following the standard transformer block structure:
    1. Self-attention with residual connection
    2. MLP with residual connection
    """
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0, 
                 causal: bool = False, use_flash_attn: bool = True):
        super().__init__()

        # layer normalization, then multi-head self attention
        self.norm1 = RMSNorm(dim)
        self.attn = MultiheadAttentionFlash(
            dim, n_heads, 
            dropout=dropout, 
            causal=causal, 
            use_flash_attn=use_flash_attn)
        
        # layer normalization, then MLP
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, hidden_mult=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor):
        
        x = x + self.attn(self.norm1(x)) # Pre-norm residual connection for attention
        x = x + self.mlp(self.norm2(x))  # Pre-norm residual connection for MLP
        
        return x

class TinyDiT(nn.Module):
    """
    Minimal DiT-style transformer model for test runs.
    
    Processes sequences of embeddings and can be used for various sequence 
    modeling tasks like text generation, image generation, etc.
    """
    def __init__(self, dim: int, n_heads: int, depth: int, mlp_ratio: int = 4, dropout: float = 0.0, causal: bool = False, use_flash_attn: bool = True):
        super().__init__()
        
        # stack of transformer blocks
        self.layers = nn.ModuleList([
            TinyDiTBlock(dim, n_heads, mlp_ratio, dropout, causal, use_flash_attn) 
            for _ in range(depth)])
        
        # final layer normalization
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor):
        for blk in self.layers:
            x = blk(x)

        return self.norm(x)