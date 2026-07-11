"""Bidirectional multi-head self-attention for the encoder.

No causal masking is applied — every position attends to every non-padded position, which is
what makes the encoder bidirectional. An optional PyTorch scaled-dot-product-attention (SDPA)
path is available and produces equivalent results; the explicit softmax path is the tested
default because it is easy to inspect.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import ModelConfig


def build_extended_attention_mask(
    attention_mask: Optional[torch.Tensor], dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Convert a [batch, seq] 1/0 mask into an additive [batch, 1, 1, seq] bias.

    Kept positions -> 0.0, padded positions -> a large negative number so their softmax
    weight vanishes. Returns ``None`` when no mask is supplied.
    """
    if attention_mask is None:
        return None
    ext = attention_mask[:, None, None, :].to(dtype=dtype)
    min_val = torch.finfo(dtype).min
    return (1.0 - ext) * min_val


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention (query/key/value/output projections)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.all_head_size = self.num_heads * self.head_dim
        self.use_sdpa = config.use_sdpa
        self.dropout_p = config.attention_probs_dropout_prob

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(self.dropout_p)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        b, _, s, _ = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(b, s, self.all_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        extended_attention_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self._split_heads(self.query(hidden_states))
        k = self._split_heads(self.key(hidden_states))
        v = self._split_heads(self.value(hidden_states))

        if self.use_sdpa and not return_probs:
            # F.sdpa expects an additive float mask broadcastable to [b, heads, seq, seq].
            attn_mask = extended_attention_mask
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
            probs = None
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if extended_attention_mask is not None:
                scores = scores + extended_attention_mask
            probs = torch.softmax(scores, dim=-1)
            probs = self.attn_dropout(probs)
            context = torch.matmul(probs, v)

        context = self._merge_heads(context)
        out = self.out_dropout(self.out_proj(context))
        return out, (probs if return_probs else None)
