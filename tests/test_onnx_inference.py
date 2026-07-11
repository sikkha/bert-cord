"""ONNX Runtime inference + PyTorch parity tests (tiny model). Skips if ORT absent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from coordinator_bert.configuration import ModelConfig
from coordinator_bert.model import BertForMaskedLM
from coordinator_bert import onnx_export as ox

ATOL = 2e-3


def _tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                       num_attention_heads=4, intermediate_size=64,
                       max_position_embeddings=64, type_vocab_size=2)


@pytest.fixture(scope="module")
def model_and_session(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(1)
    model = BertForMaskedLM(_tiny_config()).eval()
    path = str(tmp_path_factory.mktemp("ort") / "tiny.onnx")
    ox.export_to_onnx(model, path, sequence_length=16, opset=ox.DEFAULT_OPSET, batch_size=1)
    session = ox.create_ort_session(path)
    return model, session, path


# --------------------------------------------------------------------------------------- #
# Pure helpers (no ONNX)
# --------------------------------------------------------------------------------------- #
def test_compare_logits_detects_nan():
    a = np.zeros((1, 2, 3))
    b = a.copy()
    b[0, 0, 0] = np.nan
    stats = ox.compare_logits(a, b)
    assert stats["any_nan"] is True


def test_topk_agreement_perfect_and_partial():
    a = np.array([[[0.1, 0.9, 0.2, 0.0]]])
    assert ox.topk_agreement(a, a.copy(), [(0, 0)], k=2) == 1.0
    b = a.copy()
    b[0, 0] = np.array([0.9, 0.1, 0.8, 0.7])  # different top-2
    assert ox.topk_agreement(a, b, [(0, 0)], k=2) == 0.0


# --------------------------------------------------------------------------------------- #
# ONNX Runtime execution
# --------------------------------------------------------------------------------------- #
def test_ort_inference_runs_and_shapes(model_and_session):
    _, session, _ = model_and_session
    assert "CPUExecutionProvider" in session.get_providers()
    assert set(ox.ort_input_names(session)) == set(ox.INPUT_NAMES)
    ii, am, tt = ox.example_inputs(2, 12, 128, seed=5)
    out = ox.run_onnx_logits(session, ii, am, tt)
    assert out.shape == (2, 12, 128)
    assert not np.isnan(out).any() and not np.isinf(out).any()


def test_dynamic_batch_and_sequence(model_and_session):
    # Exported at batch=1, seq=16; run at different batch AND seq.
    _, session, _ = model_and_session
    for b, s in [(1, 16), (3, 24), (2, 8)]:
        ii, am, tt = ox.example_inputs(b, s, 128, seed=b + s)
        out = ox.run_onnx_logits(session, ii, am, tt)
        assert out.shape == (b, s, 128)


def test_pytorch_onnx_parity_multiple_shapes(model_and_session):
    model, session, _ = model_and_session
    for b, s, pad in [(1, 16, 0), (2, 24, 0), (3, 20, 2)]:  # incl. padded (attention mask)
        ii, am, tt = ox.example_inputs(b, s, 128, seed=b * 10 + s, pad_last=pad)
        ref = ox.torch_reference_logits(model, ii, am, tt)
        got = ox.run_onnx_logits(session, ii, am, tt)
        stats = ox.compare_logits(ref, got)
        assert stats["shapes_match"] and stats["shape_a"] == (b, s, 128)
        assert stats["max_abs_diff"] <= ATOL, (b, s, pad, stats["max_abs_diff"])
        assert not stats["any_nan"] and not stats["any_inf"]
        positions = [(r, c) for r in range(b) for c in (1, s // 2, s - 2)]
        assert ox.topk_agreement(ref, got, positions, k=5) == 1.0


def test_attention_mask_changes_output(model_and_session):
    # Masking out trailing tokens should change unpadded-position logits vs. no mask.
    model, session, _ = model_and_session
    ii, am_full, tt = ox.example_inputs(1, 16, 128, seed=7)
    am_pad = am_full.clone()
    am_pad[:, -4:] = 0
    out_full = ox.run_onnx_logits(session, ii, am_full, tt)
    out_pad = ox.run_onnx_logits(session, ii, am_pad, tt)
    assert np.abs(out_full[0, 2] - out_pad[0, 2]).max() > 1e-4


def test_run_onnx_logits_rejects_missing_input(model_and_session):
    _, session, _ = model_and_session
    # Feeding a wrong-rank input should raise from ONNX Runtime.
    with pytest.raises(Exception):
        bad = np.zeros((16,), dtype=np.int64)  # 1-D, not [batch, seq]
        session.run(["logits"], {"input_ids": bad,
                                 "attention_mask": bad, "token_type_ids": bad})


# --------------------------------------------------------------------------------------- #
# No regression to existing PyTorch inference
# --------------------------------------------------------------------------------------- #
def test_pytorch_predict_still_works():
    from coordinator_bert.inference import predict_masked_topk
    model = BertForMaskedLM(_tiny_config()).eval()
    seq = torch.tensor([1, 7, 3, 9, 3, 2])  # id 3 == mask token in SpecialTokens
    pos, ids, probs = predict_masked_topk(model, seq, mask_token_id=3, k=5)
    assert pos.shape[1] == 2 and ids.shape[1] == 5
