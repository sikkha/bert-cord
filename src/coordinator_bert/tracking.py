"""Optional experiment tracking — a tiny backend-agnostic interface.

Two backends: ``NullTracker`` (default, a pure no-op) and ``WandbTracker`` (Weights & Biases).
W&B is **never** required: nothing here imports ``wandb`` at module load; it is imported lazily
only when the W&B backend is actually selected. Tracking never changes training mathematics and
never replaces the project's local JSONL metrics + curve analysis — it is an *addition*.

Secrets are never logged: config/summary payloads are redacted of any key that looks like an
API key/token/password before being sent to a backend. Online mode uses W&B's normal login /
``WANDB_API_KEY`` mechanism; offline mode requires no authentication and no network.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional


class TrackingError(RuntimeError):
    """Raised when a tracking backend is selected but cannot be used (clear + actionable)."""


# Substrings that mark a config/summary key as sensitive — never forwarded to a backend.
_SECRET_MARKERS = ("api_key", "apikey", "api-key", "token", "secret", "password", "passwd",
                   "authorization", "auth_token", "access_key", "private_key", "credential",
                   "wandb_api_key")


def redact(obj: Any) -> Any:
    """Recursively drop keys whose name looks sensitive; leave everything else intact."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and any(m in k.lower() for m in _SECRET_MARKERS):
                out[k] = "***REDACTED***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def make_run_name(model_tag: str, platform_tag: str, experiment_tag: str,
                  timestamp: Optional[str] = None) -> str:
    """Deterministic, meaningful run name: <model>-<platform>-<experiment>-<timestamp>."""
    ts = timestamp or time.strftime("%Y%m%d-%H%M%S")
    parts = [p for p in (model_tag, platform_tag, experiment_tag, ts) if p]
    return "-".join(str(p) for p in parts)


# --------------------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------------------- #
class BaseTracker:
    backend = "none"

    def init_run(self, *, config: dict, run_name: str, project: Optional[str] = None,
                 entity: Optional[str] = None, group: Optional[str] = None,
                 job_type: Optional[str] = None, tags: Optional[list] = None,
                 notes: Optional[str] = None, mode: str = "offline",
                 dir: Optional[str] = None) -> None:
        ...

    def log_metrics(self, metrics: dict, step: int) -> None:
        ...

    def log_summary(self, summary: dict) -> None:
        ...

    def log_file(self, path: str, name: Optional[str] = None) -> None:
        ...

    def log_artifact(self, path: str, name: str, artifact_type: str) -> None:
        ...

    def finish(self, exit_code: int = 0) -> None:
        ...

    @property
    def is_active(self) -> bool:
        return False

    @property
    def run_dir(self) -> Optional[str]:
        return None

    @property
    def sync_command(self) -> Optional[str]:
        return None


class NullTracker(BaseTracker):
    """Does nothing. The default backend; keeps training identical to no-tracking."""

    backend = "none"

    def init_run(self, **kwargs) -> None:  # noqa: D401
        return None


class WandbTracker(BaseTracker):
    """Weights & Biases backend. Imports ``wandb`` lazily, on init only."""

    backend = "wandb"

    def __init__(self) -> None:
        self._wandb = None
        self._run = None
        self._mode = "offline"

    def init_run(self, *, config: dict, run_name: str, project: Optional[str] = None,
                 entity: Optional[str] = None, group: Optional[str] = None,
                 job_type: Optional[str] = None, tags: Optional[list] = None,
                 notes: Optional[str] = None, mode: str = "offline",
                 dir: Optional[str] = None) -> None:
        self._wandb = _import_wandb()
        self._mode = mode
        if dir:
            os.makedirs(dir, exist_ok=True)
        # Offline mode must never require auth or network.
        if mode == "offline":
            os.environ.setdefault("WANDB_MODE", "offline")
        self._run = self._wandb.init(
            project=project, entity=entity, name=run_name, group=group, job_type=job_type,
            tags=list(tags or []), notes=notes, mode=mode, dir=dir,
            config=redact(config), reinit=True, settings=self._wandb.Settings(silent=True),
        )

    def log_metrics(self, metrics: dict, step: int) -> None:
        if self._run is not None:
            self._run.log(dict(metrics), step=int(step))

    def log_summary(self, summary: dict) -> None:
        if self._run is not None:
            for k, v in redact(summary).items():
                self._run.summary[k] = v

    def log_file(self, path: str, name: Optional[str] = None) -> None:
        if self._run is not None and os.path.exists(path):
            try:
                self._run.save(path, policy="now")
            except Exception:  # noqa: BLE001
                pass

    def log_artifact(self, path: str, name: str, artifact_type: str) -> None:
        if self._run is None or not os.path.exists(path):
            return
        art = self._wandb.Artifact(name=name, type=artifact_type)
        if os.path.isdir(path):
            art.add_dir(path)
        else:
            art.add_file(path)
        self._run.log_artifact(art)

    def finish(self, exit_code: int = 0) -> None:
        if self._run is not None:
            try:
                self._run.finish(exit_code=exit_code)
            finally:
                self._run = None

    @property
    def is_active(self) -> bool:
        return self._run is not None

    @property
    def run_dir(self) -> Optional[str]:
        if self._run is None:
            return None
        # wandb run.dir is the '.../files' subdir; the syncable dir is its parent.
        try:
            return os.path.dirname(self._run.dir)
        except Exception:  # noqa: BLE001
            return None

    @property
    def sync_command(self) -> Optional[str]:
        rd = self.run_dir
        if rd and self._mode == "offline":
            return f"wandb sync {rd}"
        return None


def _import_wandb():
    try:
        import wandb  # noqa: PLC0415  (intentional lazy import)
        return wandb
    except Exception as e:  # noqa: BLE001
        raise TrackingError(
            "Tracking backend 'wandb' was selected but the 'wandb' package is not installed. "
            "Install it with:  python -m pip install -e \".[wandb]\"  "
            "(or set tracking.backend: none)."
        ) from e


def build_tracker(tracking_cfg) -> BaseTracker:
    """Construct a tracker from a TrackingConfig. Default/none -> NullTracker.

    Selecting 'wandb' while the package is absent fails fast with an actionable TrackingError.
    """
    backend = (getattr(tracking_cfg, "backend", "none") or "none").lower()
    if backend in ("none", "null", ""):
        return NullTracker()
    if backend == "wandb":
        _import_wandb()  # fail fast with a clear message if unavailable
        return WandbTracker()
    raise TrackingError(f"unknown tracking backend {backend!r} (use 'none' or 'wandb').")
