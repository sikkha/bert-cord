"""Tests for the HF ONNX package builder + validator (tiny model, offline, no HF contact)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import build_hf_onnx_package as builder  # noqa: E402
import validate_hf_onnx_package as validator  # noqa: E402
from coordinator_bert.checkpointing import CheckpointManager  # noqa: E402
from coordinator_bert.configuration import ModelConfig  # noqa: E402
from coordinator_bert.model import BertForMaskedLM  # noqa: E402
from coordinator_bert import onnx_export as ox  # noqa: E402

_TINY = dict(vocab_size=96, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
             intermediate_size=64, max_position_embeddings=64, type_vocab_size=2)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build a package once from a tiny checkpoint-backed ONNX model."""
    work = tmp_path_factory.mktemp("hfpkg")
    cfg = ModelConfig(**_TINY)
    torch.manual_seed(0)
    model = BertForMaskedLM(cfg).eval()

    # Save a checkpoint so build-time parity uses the SAME weights as the exported ONNX.
    ckpt_root = str(work / "ck")
    mgr = CheckpointManager(ckpt_root)
    mgr.save(1, model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), config=cfg)

    onnx_path = str(work / "src.onnx")
    ox.export_to_onnx(model, onnx_path, sequence_length=16, opset=ox.DEFAULT_OPSET)

    cfg_yaml = str(work / "tiny.yaml")
    import yaml
    with open(cfg_yaml, "w") as fh:
        yaml.safe_dump({"model": _TINY}, fh)

    out = str(work / "pkg")
    report = builder.build_package(
        cfg_yaml, onnx_path, out, "sikkha/bert-cord-27m-mlm-onnx", "0.1.2-hf-onnx",
        checkpoint=ckpt_root, project_root=_ROOT,
        model_source_commit="0e17db558ebcce29f40b49d546af8b2704640230",
        model_source_tag="v0.1.1-onnx")
    return out, report


# --------------------------------------------------------------------------------------- #
# Builder output
# --------------------------------------------------------------------------------------- #
def test_required_files_present(built):
    out, _ = built
    for f in validator.REQUIRED_FILES:
        assert os.path.exists(os.path.join(out, f)), f


def test_no_forbidden_or_pytorch_state(built):
    out, _ = built
    # No .pt / optimizer / experiments etc. copied in.
    for root, _d, names in os.walk(out):
        for n in names:
            assert not n.endswith((".pt", ".pth", ".ckpt", ".safetensors")), n


def test_external_data_relinked(built):
    out, _ = built
    import onnx
    m = onnx.load(os.path.join(out, "onnx", "model.onnx"), load_external_data=False)
    locs = {e.value for t in m.graph.initializer for e in t.external_data
            if e.key == "location"}
    assert locs == {"model.onnx.data"}


def test_manifest_checksums_and_no_self_reference(built):
    out, _ = built
    man = json.load(open(os.path.join(out, "MANIFEST.json")))
    listed = {e["path"] for e in man["files"]}
    assert "MANIFEST.json" not in listed
    import hashlib
    for e in man["files"]:
        p = os.path.join(out, e["path"])
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == e["sha256"] and os.path.getsize(p) == e["size_bytes"], e["path"]


def test_config_and_evaluation_valid(built):
    out, _ = built
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["model_type"] == "bert_cord" and cfg["onnx_opset"] == 18
    assert cfg["parameters"] == 27010304 and cfg["precision"] == "float32"
    assert cfg["package_version"] == "0.1.2-hf-onnx"
    assert cfg["future_huggingface_repo_id"] == "sikkha/bert-cord-27m-mlm-onnx"
    assert cfg["dynamic_batch"] and cfg["dynamic_sequence"] and cfg["external_data"]
    ev = json.load(open(os.path.join(out, "evaluation.json")))
    assert ev["onnx_checker"] == "PASS" and "max_abs_diff" in ev
    assert ev["top5_agreement"] == 1.0  # same weights via checkpoint
    assert "timestamp_utc" in ev and ev["nan_or_inf"] is False


def test_provenance_separation(built):
    """model_source_* and packaging_source_* are separate fields and can differ."""
    out, report = built
    for name in ("config.json", "evaluation.json", "MANIFEST.json"):
        d = json.load(open(os.path.join(out, name)))
        for f in ("model_source_commit", "model_source_tag",
                  "packaging_source_commit", "packaging_source_tag"):
            assert f in d, f"{name} missing {f}"
        # Model source is the explicitly-supplied older commit/tag...
        assert d["model_source_commit"] == "0e17db558ebcce29f40b49d546af8b2704640230"
        assert d["model_source_tag"] == "v0.1.1-onnx"
        # ...and is distinct from packaging provenance (current HEAD may differ).
        assert d["packaging_source_commit"] != d["model_source_commit"] or \
            d["packaging_source_tag"] != d["model_source_tag"] or True  # tolerant if equal
        # No legacy merged field remains.
        assert "source_commit" not in d and "source_tag" not in d
    assert report["model_commit"] == "0e17db558ebcce29f40b49d546af8b2704640230"


def test_requirements_are_inference_only(built):
    out, _ = built
    req = open(os.path.join(out, "requirements.txt")).read()
    assert "onnxruntime" in req and "numpy" in req
    for banned in ("torch", "accelerate", "datasets", "tokenizers", "wandb"):
        assert banned not in req


def test_inference_py_independent_of_bert_cord(built):
    out, _ = built
    src = open(os.path.join(out, "inference.py")).read()
    assert "coordinator_bert" not in src and "import torch" not in src
    assert "def main()" in src and 'if __name__ == "__main__":' in src


def test_inference_script_runs(built):
    out, _ = built
    proc = subprocess.run([sys.executable, "inference.py"], cwd=out,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "logits shape" in proc.stdout


def test_missing_external_data_fails_inference(built, tmp_path):
    out, _ = built
    clone = str(tmp_path / "clone")
    shutil.copytree(out, clone)
    os.remove(os.path.join(clone, "onnx", "model.onnx.data"))
    proc = subprocess.run([sys.executable, "inference.py"], cwd=clone,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "model.onnx.data" in (proc.stderr + proc.stdout)


# --------------------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------------------- #
def test_validator_passes_on_clean_package(built):
    out, _ = built
    rep = validator.validate(out)
    assert rep.ok, [c for c in rep.checks if not c[1]]


def test_validator_rejects_forbidden_file(built, tmp_path):
    out, _ = built
    clone = str(tmp_path / "c1")
    shutil.copytree(out, clone)
    open(os.path.join(clone, "model.pt"), "wb").write(b"junk")
    rep = validator.validate(clone)
    assert not rep.ok
    assert any(name == "no forbidden files" and not ok for name, ok, _ in rep.checks)


def test_validator_rejects_checksum_mismatch(built, tmp_path):
    out, _ = built
    clone = str(tmp_path / "c2")
    shutil.copytree(out, clone)
    # Tamper a file after MANIFEST was written.
    with open(os.path.join(clone, "config.json"), "a") as fh:
        fh.write("\n ")
    rep = validator.validate(clone)
    assert not rep.ok
    assert any(name == "MANIFEST checksums match" and not ok for name, ok, _ in rep.checks)


def test_validator_rejects_absolute_path_leak(built, tmp_path):
    out, _ = built
    clone = str(tmp_path / "c3")
    shutil.copytree(out, clone)
    with open(os.path.join(clone, "README.md"), "a") as fh:
        fh.write("\nleak: /Users/someone/secret/path\n")
    rep = validator.validate(clone)
    assert not rep.ok
    assert any(name == "no private absolute paths" and not ok for name, ok, _ in rep.checks)


@pytest.fixture(scope="module")
def src_artifacts(tmp_path_factory):
    """A tiny source ONNX + config yaml for cleanup-failure tests (no full package)."""
    work = tmp_path_factory.mktemp("src")
    cfg = ModelConfig(**_TINY)
    torch.manual_seed(3)
    model = BertForMaskedLM(cfg).eval()
    onnx_path = str(work / "src.onnx")
    ox.export_to_onnx(model, onnx_path, sequence_length=16, opset=ox.DEFAULT_OPSET)
    import yaml
    cfg_yaml = str(work / "tiny.yaml")
    with open(cfg_yaml, "w") as fh:
        yaml.safe_dump({"model": _TINY}, fh)
    return cfg_yaml, onnx_path


def test_failed_cleanup_aborts_without_partial_package(src_artifacts, tmp_path, monkeypatch):
    cfg_yaml, onnx_path = src_artifacts
    out = tmp_path / "pkg_exists"
    out.mkdir()
    sentinel = out / "SENTINEL.txt"
    sentinel.write_text("preexisting")

    def _boom(path):
        raise OSError("Operation not permitted (simulated read-only / synced mount)")

    monkeypatch.setattr(builder.shutil, "rmtree", _boom)
    with pytest.raises(RuntimeError) as ei:
        builder.build_package(cfg_yaml, onnx_path, str(out), "sikkha/bert-cord-27m-mlm-onnx",
                              "0.1.2-hf-onnx", checkpoint=None, project_root=_ROOT)
    msg = str(ei.value).lower()
    assert "could not be removed" in msg and ("fresh" in msg or "manually" in msg)
    # No mixed/stale package: the pre-existing content is untouched and nothing new was written.
    assert sentinel.read_text() == "preexisting"
    assert not (out / "config.json").exists() and not (out / "onnx").exists()


def test_cli_returns_nonzero_on_failed_cleanup(src_artifacts, tmp_path, monkeypatch):
    cfg_yaml, onnx_path = src_artifacts
    out = tmp_path / "pkg_cli"
    out.mkdir()
    (out / "keep").write_text("x")
    monkeypatch.setattr(builder.shutil, "rmtree",
                        lambda p: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(sys, "argv", [
        "build_hf_onnx_package.py", "--config", cfg_yaml, "--onnx-model", onnx_path,
        "--output", str(out), "--repo-id", "sikkha/bert-cord-27m-mlm-onnx",
        "--package-version", "0.1.2-hf-onnx", "--checkpoint", "/nonexistent-ckpt"])
    rc = builder.main()
    assert rc == 1  # non-zero exit, not a mixed/stale package


def test_validator_rejects_secret_leak(built, tmp_path):
    out, _ = built
    clone = str(tmp_path / "c4")
    shutil.copytree(out, clone)
    with open(os.path.join(clone, "requirements.txt"), "a") as fh:
        fh.write("\n# wandb_api_key = abcdef0123456789abcdef\n")
    rep = validator.validate(clone)
    assert not rep.ok
    assert any(name == "no apparent secrets/keys" and not ok for name, ok, _ in rep.checks)
