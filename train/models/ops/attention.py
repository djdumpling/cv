# FlashAttention-3 for H100 training

from typing import Optional, Tuple
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as _fa_func
    _HAS_FLASH_ATTN = True
except Exception as e:
    _HAS_FLASH_ATTN = False

class MultiheadAttentionFlash(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False, use_flash_attn: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._use_fa = bool(use_flash_attn and _HAS_FLASH_ATTN)

        # Enable PyTorch SDPA flash by default (as fallback)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    @staticmethod
    def _rearrange_qkv(x: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, C = x.shape
        # C = 3 * embed_dim, so head_dim = embed_dim // num_heads = C // (3 * num_heads)
        head_dim = C // (3 * num_heads)
        qkv = x.view(B, L, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v  # [B, H, L, D]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = self._rearrange_qkv(qkv, self.num_heads)  # [B, H, L, D]
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        used_fa = False
        if self._use_fa and q.is_cuda:
            # flash_attn_func expects [B, L, H, D]
            q_f = q.permute(0, 2, 1, 3)  # [B, L, H, D]
            k_f = k.permute(0, 2, 1, 3)
            v_f = v.permute(0, 2, 1, 3)
            # constraints for FA kernels
            if q_f.dtype in (torch.float16, torch.bfloat16) and (q_f.shape[-1] % 8 == 0):
                out = _fa_func(q_f, k_f, v_f, dropout_p=self.dropout if self.training else 0.0,
                               softmax_scale=None, causal=self.causal)
                out = out.permute(0, 2, 1, 3).contiguous()  # back to [B, H, L, D]
                out = out.reshape(B, L, C)
                used_fa = True
            else:
                warnings.warn("FlashAttention available but head_dim not multiple of 8 or dtype not fp16/bf16; falling back to SDPA.")

        if not used_fa:
            # use PyTorch SDPA with flash backend preferred; PyTorch will pick best available.
            q_s = q.permute(0, 2, 1, 3)  # [B, L, H, D]
            k_s = k.permute(0, 2, 1, 3)
            v_s = v.permute(0, 2, 1, 3)
            q_s = q_s.reshape(B, L, C).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,D]
            k_s = k_s.reshape(B, L, C).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v_s = v_s.reshape(B, L, C).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(
                    q_s, k_s, v_s,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=self.causal
                )  # [B,H,L,D]
            out = out.transpose(1, 2).contiguous().view(B, L, C)

        return self.proj(out)

    def report_backend(self) -> str:
        if self._use_fa:
            return "flash-attn (FA-3)"
        return "pytorch-sdpa (flash/mem_efficient/math auto)"