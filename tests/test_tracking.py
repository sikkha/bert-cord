"""Tests for optional W&B tracking. The wandb SDK is mocked — no network, no wandb.ai."""

from __future__ import annotations

import os
import sys
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from coordinator_bert import tracking as tk
from coordinator_bert.configuration import RunConfig, TrackingConfig


# --------------------------------------------------------------------------------------- #
# Fake wandb SDK
# --------------------------------------------------------------------------------------- #
class _FakeRun:
    def __init__(self, dir_):
        self.dir = os.path.join(dir_, "files")
        os.makedirs(self.dir, exist_ok=True)
        self.logged = []          # list of (metrics, step)
        self.summary = {}
        self.saved = []
        self.artifacts = []
        self.finished_code = None

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def save(self, path, policy=None):
        self.saved.append(path)

    def log_artifact(self, art):
        self.artifacts.append(art)

    def finish(self, exit_code=0):
        self.finished_code = exit_code


class _FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.added = []

    def add_file(self, p):
        self.added.append(("file", p))

    def add_dir(self, p):
        self.added.append(("dir", p))


def _make_fake_wandb(tmpdir):
    mod = types.ModuleType("wandb")
    state = {"init_kwargs": None, "run": None}

    def init(**kwargs):
        state["init_kwargs"] = kwargs
        run = _FakeRun(kwargs.get("dir") or tmpdir)
        state["run"] = run
        return run

    mod.init = init
    mod.Artifact = _FakeArtifact
    mod.Settings = lambda **kw: types.SimpleNamespace(**kw)
    mod._state = state
    return mod


@pytest.fixture
def fake_wandb(tmp_path, monkeypatch):
    mod = _make_fake_wandb(str(tmp_path))
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


# --------------------------------------------------------------------------------------- #
# NullTracker / build_tracker
# --------------------------------------------------------------------------------------- #
def test_null_tracker_is_noop():
    t = tk.NullTracker()
    # Every method callable and harmless.
    t.init_run(config={"a": 1}, run_name="x")
    t.log_metrics({"train/loss": 1.0}, step=1)
    t.log_summary({"final_val_loss": 0.5})
    t.log_file("/nonexistent")
    t.log_artifact("/nonexistent", "n", "config")
    t.finish()
    assert t.is_active is False and t.run_dir is None and t.sync_command is None


def test_build_tracker_default_is_null():
    assert isinstance(tk.build_tracker(TrackingConfig()), tk.NullTracker)


def test_build_tracker_wandb_absent_raises_actionable(monkeypatch):
    # Simulate wandb not being importable.
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(tk.TrackingError) as ei:
        tk.build_tracker(TrackingConfig(backend="wandb"))
    assert "pip install" in str(ei.value).lower()


def test_build_tracker_unknown_backend_raises():
    with pytest.raises(tk.TrackingError):
        tk.build_tracker(TrackingConfig(backend="mlflow"))


def test_import_does_not_pull_in_wandb_for_null(monkeypatch):
    # Building/using a NullTracker must not import wandb even if selection is 'none'.
    monkeypatch.setitem(sys.modules, "wandb", None)  # any wandb import would now fail
    t = tk.build_tracker(TrackingConfig(backend="none"))
    t.init_run(config={}, run_name="r")  # must not raise
    assert isinstance(t, tk.NullTracker)


# --------------------------------------------------------------------------------------- #
# Redaction / run name
# --------------------------------------------------------------------------------------- #
def test_redact_removes_secret_like_keys():
    d = {"lr": 1e-4, "WANDB_API_KEY": "sk-123", "nested": {"api_key": "x", "ok": 2},
         "list": [{"token": "t"}, {"fine": 1}]}
    r = tk.redact(d)
    assert r["lr"] == 1e-4
    assert r["WANDB_API_KEY"] == "***REDACTED***"
    assert r["nested"]["api_key"] == "***REDACTED***" and r["nested"]["ok"] == 2
    assert r["list"][0]["token"] == "***REDACTED***" and r["list"][1]["fine"] == 1


def test_make_run_name_format():
    name = tk.make_run_name("bert27m", "cuda", "dgx_portability", timestamp="20260711-183000")
    assert name == "bert27m-cuda-dgx_portability-20260711-183000"


# --------------------------------------------------------------------------------------- #
# WandbTracker (mocked)
# --------------------------------------------------------------------------------------- #
def test_wandb_offline_init_no_network(fake_wandb, tmp_path):
    t = tk.WandbTracker()
    t.init_run(config={"lr": 1e-4, "api_key": "SECRET"}, run_name="run-1",
               project="bert-cord", mode="offline", dir=str(tmp_path / "out"))
    kw = fake_wandb._state["init_kwargs"]
    assert kw["mode"] == "offline" and kw["name"] == "run-1" and kw["project"] == "bert-cord"
    # Secrets redacted before reaching wandb.
    assert kw["config"]["api_key"] == "***REDACTED***" and kw["config"]["lr"] == 1e-4
    assert t.is_active is True
    assert t.sync_command.startswith("wandb sync ")


def test_wandb_log_metrics_names_and_step(fake_wandb):
    t = tk.WandbTracker()
    t.init_run(config={}, run_name="r", mode="offline")
    t.log_metrics({"train/loss": 2.0, "train/learning_rate": 1e-4}, step=7)
    t.log_metrics({"eval/loss": 1.5, "eval/perplexity": 4.5}, step=10)
    logged = fake_wandb._state["run"].logged
    assert logged[0] == ({"train/loss": 2.0, "train/learning_rate": 1e-4}, 7)
    assert logged[1] == ({"eval/loss": 1.5, "eval/perplexity": 4.5}, 10)


def test_wandb_summary_redacts_secrets(fake_wandb):
    t = tk.WandbTracker()
    t.init_run(config={}, run_name="r", mode="offline")
    t.log_summary({"best_val_loss": 1.2, "api_key": "SECRET"})
    summ = fake_wandb._state["run"].summary
    assert summ["best_val_loss"] == 1.2 and summ["api_key"] == "***REDACTED***"


def test_wandb_finish_forwards_exit_code(fake_wandb):
    t = tk.WandbTracker()
    t.init_run(config={}, run_name="r", mode="offline")
    run = fake_wandb._state["run"]
    t.finish(exit_code=0)
    assert run.finished_code == 0 and t.is_active is False


def test_wandb_artifact_logging(fake_wandb, tmp_path):
    t = tk.WandbTracker()
    t.init_run(config={}, run_name="r", mode="offline")
    f = tmp_path / "resolved_config.yaml"
    f.write_text("model: {}\n")
    t.log_artifact(str(f), "resolved_config", "config")
    arts = fake_wandb._state["run"].artifacts
    assert len(arts) == 1 and arts[0].type == "config"


# --------------------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------------------- #
def test_tracking_config_defaults_are_safe():
    c = TrackingConfig()
    assert c.backend == "none"                     # default: no tracking
    assert c.mode == "offline"
    assert c.log_checkpoints is False              # checkpoint artifacts off by default
    # tags coerced to tuple from a YAML list
    c2 = TrackingConfig.from_dict({"backend": "wandb", "tags": ["a", "b"], "bogus": 1})
    assert c2.backend == "wandb" and c2.tags == ("a", "b")


def test_runconfig_has_tracking_default_none():
    cfg = RunConfig.from_dict({})
    assert cfg.tracking.backend == "none"


# --------------------------------------------------------------------------------------- #
# Trainer integration: finish always called (incl. exceptions); metric names/step correct
# --------------------------------------------------------------------------------------- #
class _SpyTracker(tk.BaseTracker):
    backend = "spy"

    def __init__(self):
        self.metrics = []
        self.summary = {}
        self.artifacts = []
        self.finished = False
        self._active = False

    def init_run(self, **kwargs):
        self._active = True
        self.init_config = kwargs.get("config")

    def log_metrics(self, metrics, step):
        self.metrics.append((dict(metrics), step))

    def log_summary(self, summary):
        self.summary.update(summary)

    def log_artifact(self, path, name, artifact_type):
        self.artifacts.append((name, artifact_type))

    def finish(self, exit_code=0):
        self.finished = True

    @property
    def is_active(self):
        return self._active


def _tiny_cfg(tmp_path):
    return RunConfig.from_dict({
        "model": {"vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 1,
                  "num_attention_heads": 4, "intermediate_size": 64,
                  "max_position_embeddings": 64, "type_vocab_size": 2},
        "train": {"max_steps": 4, "eval_every": 2, "save_every": 100, "warmup_steps": 1,
                  "per_device_batch_size": 4, "gradient_accumulation_steps": 1,
                  "max_seq_length": 16, "log_every": 1, "eval_max_batches": 2},
        "data": {"synthetic": {"num_train_examples": 32, "num_val_examples": 16,
                               "min_len": 8, "max_len": 16}},
        "runtime": {"device": "cpu"},
        "tracking": {"backend": "wandb", "mode": "offline"},  # non-none -> tracker.init_run runs
        "output": {"dir": str(tmp_path / "run"),
                   "checkpoint_dir": str(tmp_path / "run" / "checkpoints")},
    })


def test_trainer_logs_metric_names_and_calls_finish(tmp_path, monkeypatch):
    import pretrain_mlm as pm
    spy = _SpyTracker()
    monkeypatch.setattr(pm.tk, "build_tracker", lambda cfg: spy)
    pm.train(_tiny_cfg(tmp_path), resume=None, is_smoke=False,
             metrics_file=str(tmp_path / "m.jsonl"))
    assert spy.finished is True
    names = {k for m, _ in spy.metrics for k in m}
    assert "train/loss" in names and "eval/loss" in names and "system/peak_ram_mb" in names
    # Steps are integers and monotonic-ish.
    assert all(isinstance(s, int) for _, s in spy.metrics)
    # Summary has required fields and no secret keys.
    for key in ("final_global_step", "final_val_loss", "best_val_loss", "elapsed_seconds"):
        assert key in spy.summary
    assert not any("api_key" in k.lower() or "secret" in k.lower() for k in spy.summary)


def test_trainer_finish_called_on_exception(tmp_path, monkeypatch):
    import pretrain_mlm as pm
    spy = _SpyTracker()
    monkeypatch.setattr(pm.tk, "build_tracker", lambda cfg: spy)

    def _boom(*a, **k):
        raise RuntimeError("boom during eval")

    monkeypatch.setattr(pm, "evaluate", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        pm.train(_tiny_cfg(tmp_path), resume=None, is_smoke=False)
    assert spy.finished is True   # finally block guarantees finish()


def test_trainer_no_artifacts_when_backend_none(tmp_path, monkeypatch):
    import pretrain_mlm as pm
    spy = _SpyTracker()
    monkeypatch.setattr(pm.tk, "build_tracker", lambda cfg: spy)
    cfg = _tiny_cfg(tmp_path)
    from dataclasses import replace
    cfg = replace(cfg, tracking=replace(cfg.tracking, backend="none"))
    pm.train(cfg, resume=None, is_smoke=False)
    # backend none -> init_run never called -> tracker inactive -> no metrics/artifacts logged
    assert spy.is_active is False
    assert spy.metrics == [] and spy.artifacts == []
    assert spy.finished is True   # finish still always called
