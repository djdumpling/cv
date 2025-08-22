# FlashAttention-3 for H100 training
# This module implements multi-head attention using FlashAttention for optimal performance on H100 GPUs
from typing import Optional, Tuple
import torch
import torch.nn as nn
from flash_attn import flash_attn_func as _fa_func

class MultiheadAttentionFlash(nn.Module):
    """
    multi-head attention module (for H100s) using FlashAttention-3.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False, use_flash_attn: bool = True):
        super().__init__()

        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim              # total embedding dimension (1024)
        self.num_heads = num_heads              # number of attention heads (16)
        self.head_dim = embed_dim // num_heads  # dimension per head (1024/16 = 64)
        self.causal = causal                    # causal masking for autoregressive models
        self.dropout = dropout                  # dropout rate for attention weights

        # projects input to [Q, K, V] (3 * embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # self projection
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    @staticmethod
    def rearrange_qkv(x: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        rearrange tensor from [B, L, 3*embed_dim] to separate q, k, v tensors.
        """
        B, L, C = x.shape  # [batch_size, seq_length, 3 * embed_dim]
        
        # calculate head dimension; since C = 3 * embed_dim, divide by extra 3
        head_dim = C // (3 * num_heads)
        
        # [B, L, 3, num_heads, head_dim] --> [3, B, num_heads, L, head_dim]
        qkv = x.view(B, L, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        
        # get q, k, v tensors
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, L, head_dim]
        
        return q, k, v

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        
        # compute q, k, v from input and rearrange
        qkv = self.qkv(x)
        q, k, v = self.rearrange_qkv(qkv, self.num_heads)
        
        # contiguous in memory for performance
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # FlashAttention expects [B, L, H, D] format, so [B, H, L, D] --> [B, L, H, D]
        q_f = q.permute(0, 2, 1, 3)  # [B, L, H, D]
        k_f = k.permute(0, 2, 1, 3)  # [B, L, H, D]
        v_f = v.permute(0, 2, 1, 3)  # [B, L, H, D]
        
        # FlashAttention constraints: fp16/bf16 dtype AND head_dim % 8 == 0
        if q_f.dtype in (torch.float16, torch.bfloat16) and (q_f.shape[-1] % 8 == 0):
            # compute attention
            out = _fa_func(
                q_f, k_f, v_f, 
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=None,  # add back self.scale when needed
                causal=self.causal   # casual masking as needed
            )
            
            # reshape back
            out = out.reshape(B, L, C)  # [B, H, L, D] -> [B, L, embed_dim]
        else:
            # in case quantization errors
            raise ValueError(f"Requires fp16/bf16 dtype and head_dim % 8 == 0, got dtype={q_f.dtype}, head_dim={q_f.shape[-1]}")

        return self.proj(out)

    def report_backend(self) -> str:
        return "flash-attn (FA-3)"