"""Masking tests: ~15% selection rate, 80/10/10 split, special-token safety, -100 labels."""

from __future__ import annotations

import torch

from coordinator_bert.data import SpecialTokens
from coordinator_bert.masking import MLMasker


def _make_masker(mlm_prob=0.15):
    specials = SpecialTokens()
    return MLMasker(
        mask_token_id=specials.mask,
        vocab_size=1000,
        special_token_ids=specials.all_ids,
        mlm_probability=mlm_prob,
    ), specials


def _random_batch(specials, batch=64, seq=128, vocab=1000):
    # Bodies are real tokens; frame each row with [CLS] ... [SEP].
    ids = torch.randint(specials.first_real_id, vocab, (batch, seq))
    ids[:, 0] = specials.cls
    ids[:, -1] = specials.sep
    return ids


def test_selection_rate_about_15_percent():
    masker, specials = _make_masker(0.15)
    ids = _random_batch(specials, batch=128, seq=128)
    g = torch.Generator().manual_seed(123)
    out = masker(ids, generator=g)
    selected = out.labels != -100

    # Rate is measured over MASKABLE (non-special) tokens only.
    special = masker.special_tokens_mask(ids)
    maskable = int((~special).sum())
    rate = int(selected.sum()) / maskable
    assert 0.12 < rate < 0.18, f"selection rate {rate:.4f} not ~0.15 over {maskable} tokens"


def test_80_10_10_replacement_behavior():
    masker, specials = _make_masker(0.15)
    ids = _random_batch(specials, batch=256, seq=128)
    g = torch.Generator().manual_seed(7)
    out = masker(ids, generator=g)
    selected = out.labels != -100

    n_selected = int(selected.sum())
    assert n_selected > 1000  # large enough sample for stable proportions

    became_mask = (out.input_ids == specials.mask) & selected
    unchanged = (out.input_ids == ids) & selected
    # random = selected but neither [MASK] nor identical to original
    became_random = selected & ~became_mask & (out.input_ids != ids)

    p_mask = int(became_mask.sum()) / n_selected
    p_unchanged = int(unchanged.sum()) / n_selected
    p_random = int(became_random.sum()) / n_selected

    # 80/10/10 with tolerance (random can coincide with the original id ~0.1% of the time).
    assert 0.74 < p_mask < 0.86, f"mask share {p_mask:.3f}"
    assert 0.05 < p_unchanged < 0.16, f"unchanged share {p_unchanged:.3f}"
    assert 0.05 < p_random < 0.16, f"random share {p_random:.3f}"
    assert abs((p_mask + p_unchanged + p_random) - 1.0) < 0.02


def test_special_tokens_never_masked():
    masker, specials = _make_masker(0.5)  # high prob to stress-test special safety
    ids = _random_batch(specials, batch=64, seq=64)
    # Insert extra special tokens in the interior.
    ids[:, 10] = specials.sep
    ids[:, 20] = specials.cls
    g = torch.Generator().manual_seed(1)
    out = masker(ids, generator=g)

    special = masker.special_tokens_mask(ids)
    # No special position is ever selected as a label...
    assert int((out.labels[special] != -100).sum()) == 0
    # ...and no special token id was overwritten by [MASK]/random.
    assert torch.equal(out.input_ids[special], ids[special])


def test_labels_are_minus_100_for_unselected():
    masker, specials = _make_masker(0.15)
    ids = _random_batch(specials, batch=32, seq=64)
    g = torch.Generator().manual_seed(99)
    out = masker(ids, generator=g)
    selected = out.labels != -100
    # Every non-selected label is exactly -100.
    assert torch.all(out.labels[~selected] == -100)
    # Every selected label equals the ORIGINAL token id (not the corrupted input).
    assert torch.equal(out.labels[selected], ids[selected])
