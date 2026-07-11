"""Inference tests: checkpoint loading, deterministic eval, top-k shape, mask extraction."""

from __future__ import annotations

import torch

from coordinator_bert.checkpointing import save_checkpoint
from coordinator_bert.data import SpecialTokens
from coordinator_bert.inference import (
    apply_mask_at,
    find_masked_positions,
    load_model_for_inference,
    masked_accuracy_topk,
    predict_masked_topk,
    topk_predictions,
)
from coordinator_bert.model import BertForMaskedLM


def test_load_model_from_checkpoint_restores_weights(tiny_config, tmp_path):
    model = BertForMaskedLM(tiny_config)
    # Perturb weights so a fresh random init would differ.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    ckpt = str(tmp_path / "ck")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(ckpt, model=model, optimizer=opt, global_step=0, config=tiny_config)

    loaded = load_model_for_inference(tiny_config, ckpt)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert torch.equal(p1.data, p2.data), f"mismatch at {n1}"


def test_load_model_is_in_eval_mode(tiny_config):
    model = load_model_for_inference(tiny_config, checkpoint_path=None)
    assert model.training is False


def test_deterministic_inference_in_eval_mode(tiny_config):
    model = load_model_for_inference(tiny_config, checkpoint_path=None)
    ids = torch.randint(5, tiny_config.vocab_size, (2, 12))
    with torch.no_grad():
        a = model(ids)["logits"]
        b = model(ids)["logits"]
    torch.testing.assert_close(a, b)  # eval disables dropout -> identical


def test_find_masked_positions_single_and_batch():
    specials = SpecialTokens()
    # Single sequence.
    seq = torch.tensor([specials.cls, 7, specials.mask, 9, specials.sep])
    pos = find_masked_positions(seq, specials.mask)
    assert pos.tolist() == [[0, 2]]
    # Batch with two masks.
    batch = torch.tensor([
        [specials.cls, specials.mask, 8, specials.sep],
        [specials.cls, 6, specials.mask, specials.sep],
    ])
    pos = find_masked_positions(batch, specials.mask)
    assert pos.tolist() == [[0, 1], [1, 2]]


def test_apply_mask_at_returns_originals_and_does_not_mutate():
    specials = SpecialTokens()
    seq = torch.tensor([specials.cls, 11, 12, 13, specials.sep])
    masked, originals = apply_mask_at(seq, [1, 3], specials.mask)
    assert masked.tolist() == [specials.cls, specials.mask, 12, specials.mask, specials.sep]
    assert originals.tolist() == [11, 13]
    # Original tensor untouched.
    assert seq.tolist() == [specials.cls, 11, 12, 13, specials.sep]


def test_topk_predictions_shape_and_probability():
    logits = torch.randn(4, 100)  # 4 positions, vocab 100
    ids, probs = topk_predictions(logits, k=5)
    assert ids.shape == (4, 5)
    assert probs.shape == (4, 5)
    # Probabilities are in (0, 1] and sorted descending per row.
    assert torch.all(probs > 0) and torch.all(probs <= 1)
    assert torch.all(probs[:, :-1] >= probs[:, 1:])


def test_predict_masked_topk_shapes(tiny_config):
    specials = SpecialTokens()
    model = load_model_for_inference(tiny_config, checkpoint_path=None)
    seq = torch.tensor([specials.cls, 7, specials.mask, 9, specials.mask, specials.sep])
    pos, ids, probs = predict_masked_topk(model, seq, specials.mask, k=5)
    assert pos.shape == (2, 2)          # two masked positions, (row, col)
    assert ids.shape == (2, 5)          # top-5 ids per masked position
    assert probs.shape == (2, 5)
    assert pos[:, 1].tolist() == [2, 4]  # the masked column indices


def test_predict_masked_topk_empty_when_no_mask(tiny_config):
    specials = SpecialTokens()
    model = load_model_for_inference(tiny_config, checkpoint_path=None)
    seq = torch.tensor([specials.cls, 7, 8, 9, specials.sep])
    pos, ids, probs = predict_masked_topk(model, seq, specials.mask, k=5)
    assert pos.numel() == 0
    assert ids.shape == (0, 5)


def test_masked_accuracy_topk_perfect_and_ignore_index():
    # Build logits that are certain about specific tokens.
    vocab = 20
    logits = torch.full((1, 3, vocab), -10.0)
    logits[0, 0, 5] = 10.0   # position 0 -> token 5
    logits[0, 1, 7] = 10.0   # position 1 -> token 7
    logits[0, 2, 9] = 10.0   # position 2 (ignored)
    labels = torch.tensor([[5, 7, -100]])
    acc = masked_accuracy_topk(logits, labels, ks=(1, 5))
    assert acc[1] == 1.0 and acc[5] == 1.0

    # Wrong top-1 but correct within top-5.
    logits2 = torch.randn(1, 1, vocab)
    labels2 = torch.tensor([[3]])
    # Force label 3 to be the 2nd highest.
    order = logits2[0, 0].argsort(descending=True)
    logits2[0, 0, 3] = logits2[0, 0, order[0]] - 0.1
    acc2 = masked_accuracy_topk(logits2, labels2, ks=(1, 5))
    assert acc2[1] == 0.0
    assert acc2[5] == 1.0
