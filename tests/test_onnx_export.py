"""ONNX export tests (tiny model). ONNX/ORT-dependent tests skip when packages are absent."""

from __future__ import annotations

import os
import sys

import pytest
import torch

from coordinator_bert.checkpointing import CheckpointManager
from coordinator_bert.configuration import ModelConfig
from coordinator_bert.model import BertForMaskedLM, count_parameters
from coordinator_bert import onnx_export as ox


def _tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                       num_attention_heads=4, intermediate_size=64,
                       max_position_embeddings=64, type_vocab_size=2)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return BertForMaskedLM(_tiny_config()).eval()


@pytest.fixture(scope="module")
def exported(tiny_model, tmp_path_factory):
    onnx = pytest.importorskip("onnx")  # skip module if onnx missing
    pytest.importorskip("onnxruntime")
    path = str(tmp_path_factory.mktemp("onnx") / "tiny.onnx")
    meta = ox.export_to_onnx(tiny_model, path, sequence_length=16, opset=ox.DEFAULT_OPSET,
                             batch_size=1)
    return tiny_model, path, meta


# --------------------------------------------------------------------------------------- #
# Wrapper (no ONNX needed)
# --------------------------------------------------------------------------------------- #
def test_wrapper_returns_tensor(tiny_model):
    w = ox.build_inference_wrapper(tiny_model)
    ii, am, tt = ox.example_inputs(2, 12, tiny_model.config.vocab_size)
    out = w(ii, am, tt)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 12, tiny_model.config.vocab_size)


def test_wrapper_no_loss_or_probs_branch(tiny_model):
    # The wrapper must return exactly the logits tensor (not a dict / tuple).
    w = ox.build_inference_wrapper(tiny_model)
    ii, am, tt = ox.example_inputs(1, 8, tiny_model.config.vocab_size)
    out = w(ii, am, tt)
    assert out.dim() == 3 and torch.isfinite(out).all()


def test_example_inputs_are_int64_and_masked():
    ii, am, tt = ox.example_inputs(2, 10, 64, pad_token_id=0, pad_last=3)
    assert ii.dtype == torch.long and am.dtype == torch.long and tt.dtype == torch.long
    assert (am[:, -3:] == 0).all() and (am[:, :-3] == 1).all()


# --------------------------------------------------------------------------------------- #
# Missing-dependency behavior (simulate absence)
# --------------------------------------------------------------------------------------- #
def test_missing_onnx_raises_actionable(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)  # make `import onnx` fail
    with pytest.raises(ox.OnnxDependencyError) as ei:
        ox.export_to_onnx(tiny_model, str(tmp_path / "x.onnx"), sequence_length=8)
    assert "pip install" in str(ei.value).lower()


def test_missing_onnxruntime_raises_actionable(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ox.OnnxDependencyError):
        ox.create_ort_session(str(tmp_path / "nope.onnx"))


# --------------------------------------------------------------------------------------- #
# Export artifact
# --------------------------------------------------------------------------------------- #
def test_export_creates_valid_artifact(exported):
    _, path, meta = exported
    assert os.path.exists(path) and os.path.getsize(path) > 0
    assert meta["opset"] == ox.DEFAULT_OPSET
    assert meta["input_names"] == list(ox.INPUT_NAMES)
    assert meta["output_names"] == ["logits"]
    ox.check_onnx_model(path)  # structural validation passes


def test_export_param_count_matches_model(exported):
    model, _, meta = exported
    assert meta["param_count"] == count_parameters(model)["unique"]


def test_export_from_checkpoint_matches_weights(tmp_path):
    pytest.importorskip("onnx")
    cfg = _tiny_config()
    model = BertForMaskedLM(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    mgr = CheckpointManager(str(tmp_path / "ck"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr.save(5, model=model, optimizer=opt, config=cfg)

    # Export by resolving the checkpoint ROOT (latest.json) -> weights must be the saved ones.
    onnx_path = str(tmp_path / "from_ckpt.onnx")
    meta = ox.export_checkpoint_to_onnx(cfg, str(tmp_path / "ck"), onnx_path, sequence_length=8)
    assert os.path.exists(onnx_path) and meta["checkpoint"].endswith("ck")

    ort = pytest.importorskip("onnxruntime")  # noqa: F841
    sess = ox.create_ort_session(onnx_path)
    ii, am, tt = ox.example_inputs(1, 8, cfg.vocab_size, seed=3)
    ref = ox.torch_reference_logits(model, ii, am, tt)  # same (perturbed) weights
    got = ox.run_onnx_logits(sess, ii, am, tt)
    stats = ox.compare_logits(ref, got)
    assert stats["shapes_match"] and stats["max_abs_diff"] < 2e-3


def test_malformed_checkpoint_raises(tmp_path):
    # A directory without state.pt / metadata is not a valid checkpoint to load.
    cfg = _tiny_config()
    bad = tmp_path / "bad_ckpt"
    bad.mkdir()
    (bad / "state.pt").write_bytes(b"not a real torch file")
    from coordinator_bert.inference import load_model_for_inference
    with pytest.raises(Exception):
        load_model_for_inference(cfg, str(bad))
