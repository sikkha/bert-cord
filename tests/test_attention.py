"""Attention tests: shapes, mask behavior, bidirectionality, no-NaN, determinism."""

from __future__ import annotations

import torch

from coordinator_bert.attention import (
    MultiHeadSelfAttention,
    build_extended_attention_mask,
)
from coordinator_bert.model import BertModel


def test_attention_output_shape(tiny_config):
    attn = MultiHeadSelfAttention(tiny_config).eval()
    x = torch.randn(3, 7, tiny_config.hidden_size)
    out, _ = attn(x)
    assert out.shape == (3, 7, tiny_config.hidden_size)


def test_attention_returns_probs_shape(tiny_config):
    attn = MultiHeadSelfAttention(tiny_config).eval()
    x = torch.randn(2, 5, tiny_config.hidden_size)
    _, probs = attn(x, return_probs=True)
    assert probs.shape == (2, tiny_config.num_attention_heads, 5, 5)
    # Rows of the attention matrix sum to 1.
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2, tiny_config.num_attention_heads, 5))


def test_attention_mask_zeros_padded_positions(tiny_config):
    attn = MultiHeadSelfAttention(tiny_config).eval()
    x = torch.randn(1, 6, tiny_config.hidden_size)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])  # last two positions are padding
    ext = build_extended_attention_mask(mask, x.dtype)
    _, probs = attn(x, ext, return_probs=True)
    # No query should place attention weight on padded key positions.
    assert torch.allclose(probs[..., 4:], torch.zeros_like(probs[..., 4:]), atol=1e-6)


def test_attention_is_bidirectional(tiny_config):
    """A change at a later position must affect an earlier position's output.

    A causal (unidirectional) model would leave earlier positions unchanged; a bidirectional
    encoder does not, which is what we assert here.
    """
    model = BertModel(tiny_config).eval()
    ids = torch.randint(5, tiny_config.vocab_size, (1, 8))
    with torch.no_grad():
        base = model(ids)["last_hidden_state"]
        ids2 = ids.clone()
        ids2[0, -1] = (ids2[0, -1] + 7) % tiny_config.vocab_size  # perturb LAST token
        changed = model(ids2)["last_hidden_state"]
    # Output at position 0 (earlier) must move because it can attend to the last token.
    delta_first = (base[0, 0] - changed[0, 0]).abs().max()
    assert delta_first > 1e-6, "Encoder is not bidirectional: earlier token unaffected by later change."


def test_attention_no_nan(tiny_config):
    attn = MultiHeadSelfAttention(tiny_config).eval()
    x = torch.randn(4, 10, tiny_config.hidden_size)
    mask = torch.ones(4, 10, dtype=torch.long)
    ext = build_extended_attention_mask(mask, x.dtype)
    out, _ = attn(x, ext)
    assert torch.isfinite(out).all()


def test_attention_deterministic_without_dropout(tiny_config):
    attn = MultiHeadSelfAttention(tiny_config).eval()  # dropout p=0 in tiny_config + eval
    x = torch.randn(2, 6, tiny_config.hidden_size)
    out1, _ = attn(x)
    out2, _ = attn(x)
    torch.testing.assert_close(out1, out2)


def test_sdpa_matches_manual_shape(tiny_config):
    """The optional SDPA path yields the same shape and finite values as manual softmax."""
    from dataclasses import replace

    cfg_sdpa = replace(tiny_config, use_sdpa=True)
    attn = MultiHeadSelfAttention(cfg_sdpa).eval()
    x = torch.randn(2, 6, cfg_sdpa.hidden_size)
    out, probs = attn(x)
    assert out.shape == (2, 6, cfg_sdpa.hidden_size)
    assert probs is None  # SDPA path does not expose probabilities
    assert torch.isfinite(out).all()
