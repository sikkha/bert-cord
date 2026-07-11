"""Tests for immutable, checksum-verified checkpoints (CheckpointManager + verify path)."""

from __future__ import annotations

import json
import os

import pytest
import torch

from coordinator_bert.checkpointing import (
    CheckpointError,
    CheckpointManager,
    load_checkpoint,
    resolve_checkpoint_path,
    sha256_file,
)
from coordinator_bert.model import BertForMaskedLM


def _model_opt(tiny_config):
    m = BertForMaskedLM(tiny_config)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, opt


def test_atomic_save_creates_immutable_step_dir(tiny_config, tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    m, opt = _model_opt(tiny_config)
    path = mgr.save(50, model=m, optimizer=opt, config=tiny_config, precision="fp32",
                    device="cpu")
    assert os.path.basename(path) == "step_000050"
    assert os.path.exists(os.path.join(path, "state.pt"))
    assert os.path.exists(os.path.join(path, "metadata.json"))
    # No leftover temp artifacts.
    assert not any(n.endswith(".tmp") for n in os.listdir(tmp_path))


def test_metadata_records_required_fields(tiny_config, tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    m, opt = _model_opt(tiny_config)
    path = mgr.save(50, model=m, optimizer=opt, config=tiny_config, precision="bf16",
                    device="cuda")
    meta = json.load(open(os.path.join(path, "metadata.json")))
    for key in ("global_step", "git_commit", "config_hash", "param_count", "precision",
                "device", "sha256", "created_at", "torch_version"):
        assert key in meta
    assert meta["global_step"] == 50
    assert meta["precision"] == "bf16" and meta["device"] == "cuda"
    # Recorded checksum matches the actual file.
    assert meta["sha256"] == sha256_file(os.path.join(path, "state.pt"))
    assert meta["param_count"] > 0


def test_load_by_explicit_path_restores_weights_and_step(tiny_config, tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    m, opt = _model_opt(tiny_config)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.02)
    path = mgr.save(120, model=m, optimizer=opt, config=tiny_config)

    m2 = BertForMaskedLM(tiny_config)
    payload = load_checkpoint(path, model=m2, restore_rng=False, verify_checksum=True)
    assert payload["global_step"] == 120
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1.data, p2.data), n1


def test_latest_and_best_pointers(tiny_config, tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    m, opt = _model_opt(tiny_config)
    mgr.save(50, model=m, optimizer=opt, config=tiny_config)
    mgr.save(200, model=m, optimizer=opt, config=tiny_config)
    assert os.path.basename(mgr.latest_path()) == "step_000200"
    mgr.mark_best(50, metric=1.23)
    assert os.path.basename(mgr.best_path()) == "step_000050"
    ptr = mgr.read_pointer()
    assert ptr["latest"] == "step_000200" and ptr["best"] == "step_000050"
    assert ptr["best_metric"] == 1.23
    assert mgr.list_steps() == [50, 200]


def test_resolve_checkpoint_path_follows_latest(tiny_config, tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    m, opt = _model_opt(tiny_config)
    mgr.save(50, model=m, optimizer=opt, config=tiny_config)
    mgr.save(200, model=m, optimizer=opt, config=tiny_config)
    # Passing the ROOT resolves to the latest immutable step dir.
    resolved = resolve_checkpoint_path(str(tmp_path))
    assert os.path.basename(resolved) == "step_000200"
    # Passing an explicit step dir returns it unchanged.
    explicit = os.path.join(str(tmp_path), "step_000050")
    assert resolve_checkpoint_path(explicit) == explicit


def test_checksum_verification_passes_on_clean_checkpoint(tiny_config, tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    m, opt = _model_opt(tiny_config)
    path = mgr.save(10, model=m, optimizer=opt, config=tiny_config)
    m2 = BertForMaskedLM(tiny_config)
    load_checkpoint(path, model=m2, restore_rng=False, verify_checksum=True)  # no raise


def test_corrupted_checkpoint_is_rejected(tiny_config, tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    m, opt = _model_opt(tiny_config)
    path = mgr.save(10, model=m, optimizer=opt, config=tiny_config)
    # Corrupt the state file after the fact.
    with open(os.path.join(path, "state.pt"), "ab") as fh:
        fh.write(b"\x00garbage")
    m2 = BertForMaskedLM(tiny_config)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, model=m2, restore_rng=False, verify_checksum=True)


def test_resume_step_restoration_with_optimizer(tiny_config, tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    m, opt = _model_opt(tiny_config)
    ids = torch.randint(5, tiny_config.vocab_size, (2, 8))
    m(ids, labels=ids)["loss"].backward()
    opt.step()
    mgr.save(77, model=m, optimizer=opt, config=tiny_config)

    m2 = BertForMaskedLM(tiny_config)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    payload = load_checkpoint(mgr.latest_path(), model=m2, optimizer=opt2, restore_rng=False,
                              verify_checksum=True)
    assert payload["global_step"] == 77
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_verify_requested_but_missing_sha_raises(tiny_config, tmp_path):
    # A bare state.pt with no metadata sidecar cannot be checksum-verified.
    m, opt = _model_opt(tiny_config)
    from coordinator_bert.checkpointing import _atomic_torch_save
    p = str(tmp_path / "bare.pt")
    _atomic_torch_save({"model": m.state_dict(), "global_step": 1}, p)
    m2 = BertForMaskedLM(tiny_config)
    with pytest.raises(CheckpointError):
        load_checkpoint(p, model=m2, restore_rng=False, verify_checksum=True)
