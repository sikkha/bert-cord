"""Checkpoint save / resume with full training-state restoration.

A checkpoint captures everything needed to resume a run bit-for-bit (modulo backend
nondeterminism): model weights, optimizer state, LR-scheduler state, AMP scaler state, the
global step, the config dict, and RNG state for Python/NumPy/Torch (CPU + CUDA).

Reliability (Milestone 0.7 / release prep):
  * writes are **atomic** — data goes to a temp path and is renamed into place;
  * a separate ``metadata.json`` records step, git commit, config hash, parameter count,
    precision, device, timestamp and a **SHA-256** of the primary state file;
  * loads can **verify the checksum** and reject corrupted checkpoints;
  * ``CheckpointManager`` writes **immutable** ``step_XXXXXX/`` directories and a tiny
    ``latest.json`` pointer (and a ``best`` pointer) so nothing needs to duplicate a
    300+ MB state file for "last"/"best".

The legacy ``save_checkpoint`` / ``load_checkpoint`` functions are preserved for backward
compatibility (now with atomic writes + optional checksum metadata).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch

FORMAT_VERSION = 2


# --------------------------------------------------------------------------------------- #
# RNG state
# --------------------------------------------------------------------------------------- #
def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["torch_cuda"]])
        except (RuntimeError, ValueError):
            pass  # different device count on resume; skip CUDA RNG restore


def _as_byte_tensor(t: Any) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.cpu().to(torch.uint8)
    return torch.tensor(t, dtype=torch.uint8)


# --------------------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------------------- #
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(config: Any) -> Optional[str]:
    d = _config_to_dict(config)
    if d is None:
        return None
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__)))))
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _config_to_dict(config: Any) -> Any:
    if config is None:
        return None
    try:
        return asdict(config)
    except TypeError:
        return config


def _atomic_torch_save(payload: dict, final_path: str) -> None:
    """Write a torch payload to ``final_path`` atomically (temp file + os.replace)."""
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    tmp = final_path + ".tmp"
    torch.save(payload, tmp)
    try:
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    except Exception:  # noqa: BLE001
        pass
    os.replace(tmp, final_path)  # atomic on POSIX/NTFS within a filesystem


def _atomic_write_text(text: str, final_path: str) -> None:
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    tmp = final_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, final_path)


# --------------------------------------------------------------------------------------- #
# Legacy single-directory API (backward compatible, now atomic + checksummed)
# --------------------------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    global_step: int = 0,
    config: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
    save_rng: bool = True,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> str:
    """Save a checkpoint directory with an atomic state.pt + metadata.json (with SHA-256)."""
    os.makedirs(path, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "config": _config_to_dict(config),
        "metadata": metadata or {},
    }
    if save_rng:
        payload["rng_state"] = _rng_state()

    state_path = os.path.join(path, "state.pt")
    _atomic_torch_save(payload, state_path)

    meta = _build_metadata(model, global_step, config, metadata, precision, device, state_path)
    # Legacy consumers read meta.json; also write metadata.json (new canonical name).
    text = json.dumps(meta, indent=2, default=str)
    _atomic_write_text(text, os.path.join(path, "meta.json"))
    _atomic_write_text(text, os.path.join(path, "metadata.json"))
    return path


def _build_metadata(model, global_step, config, metadata, precision, device,
                    state_path) -> dict:
    try:
        from .model import count_parameters
        param_count = count_parameters(model)["unique"]
    except Exception:  # noqa: BLE001
        param_count = sum(p.numel() for p in model.parameters())
    return {
        "format_version": FORMAT_VERSION,
        "global_step": global_step,
        "git_commit": _git_commit(),
        "config": _config_to_dict(config),
        "config_hash": config_hash(config),
        "param_count": int(param_count),
        "precision": precision,
        "device": device,
        "sha256": sha256_file(state_path),
        "state_bytes": os.path.getsize(state_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
        "metadata": metadata or {},
    }


class CheckpointError(RuntimeError):
    pass


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Optional[Any] = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
    verify_checksum: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint and restore in-place. Optionally verify the SHA-256 first.

    ``path`` may be a checkpoint directory (containing state.pt) or a direct state.pt path.
    With ``verify_checksum=True`` and an available metadata sidecar, a mismatch raises
    ``CheckpointError`` before any state is loaded.
    """
    if os.path.isdir(path):
        state_file = os.path.join(path, "state.pt")
        meta = _read_metadata_sidecar(path)
    else:
        state_file = path
        meta = _read_metadata_sidecar(os.path.dirname(path))

    if verify_checksum:
        expected = (meta or {}).get("sha256")
        if not expected:
            raise CheckpointError(f"checksum verification requested but no sha256 in "
                                  f"metadata for {path}")
        actual = sha256_file(state_file)
        if actual != expected:
            raise CheckpointError(
                f"checkpoint checksum mismatch for {state_file}: expected {expected[:12]}…, "
                f"got {actual[:12]}… (corrupted or tampered).")

    payload = torch.load(state_file, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng_state") is not None:
        _set_rng_state(payload["rng_state"])
    return payload


def _read_metadata_sidecar(dir_path: str) -> Optional[dict]:
    for name in ("metadata.json", "meta.json"):
        p = os.path.join(dir_path, name)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:  # noqa: BLE001
                return None
    return None


# --------------------------------------------------------------------------------------- #
# Immutable checkpoint manager (step_XXXXXX/ dirs + latest.json pointer)
# --------------------------------------------------------------------------------------- #
class CheckpointManager:
    """Writes immutable per-step checkpoints and maintains a tiny latest/best pointer file.

    Layout under ``root``::

        step_000050/  (state.pt + metadata.json)
        step_000200/
        latest.json   {"latest": "step_000200", "best": "step_000050", "best_metric": ...}

    "best" is a pointer, never a duplicated copy of the multi-hundred-MB state file.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    @staticmethod
    def step_dirname(step: int) -> str:
        return f"step_{step:06d}"

    def save(self, step: int, *, model, optimizer, scheduler=None, scaler=None,
             config=None, metadata=None, precision=None, device=None,
             save_rng: bool = True) -> str:
        """Atomically write an immutable step_XXXXXX/ checkpoint; update latest pointer."""
        dirname = self.step_dirname(step)
        final_dir = os.path.join(self.root, dirname)
        tmp_dir = final_dir + ".tmp"
        if os.path.isdir(tmp_dir):
            _rmtree_quiet(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        payload = {
            "format_version": FORMAT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": step,
            "config": _config_to_dict(config),
            "metadata": metadata or {},
        }
        if save_rng:
            payload["rng_state"] = _rng_state()

        state_path = os.path.join(tmp_dir, "state.pt")
        torch.save(payload, state_path)  # inside tmp dir; whole dir renamed atomically below
        meta = _build_metadata(model, step, config, metadata, precision, device, state_path)
        with open(os.path.join(tmp_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

        # Immutable: never overwrite an existing committed step dir.
        if os.path.isdir(final_dir):
            _rmtree_quiet(tmp_dir)
        else:
            os.replace(tmp_dir, final_dir)  # atomic directory publish
        self._update_pointer(latest=dirname)
        return final_dir

    def mark_best(self, step: int, metric: Optional[float] = None) -> None:
        """Point 'best' at an existing step checkpoint (no data duplication)."""
        self._update_pointer(best=self.step_dirname(step), best_metric=metric)

    def _update_pointer(self, latest=None, best=None, best_metric=None) -> None:
        pointer = self.read_pointer() or {}
        if latest is not None:
            pointer["latest"] = latest
        if best is not None:
            pointer["best"] = best
        if best_metric is not None:
            pointer["best_metric"] = best_metric
        pointer["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_write_text(json.dumps(pointer, indent=2), os.path.join(self.root, "latest.json"))

    def read_pointer(self) -> Optional[dict]:
        p = os.path.join(self.root, "latest.json")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return None

    def latest_path(self) -> Optional[str]:
        ptr = self.read_pointer()
        if ptr and ptr.get("latest"):
            cand = os.path.join(self.root, ptr["latest"])
            return cand if os.path.isdir(cand) else None
        return None

    def best_path(self) -> Optional[str]:
        ptr = self.read_pointer()
        if ptr and ptr.get("best"):
            cand = os.path.join(self.root, ptr["best"])
            return cand if os.path.isdir(cand) else None
        return None

    def list_steps(self) -> list:
        out = []
        for name in os.listdir(self.root):
            if name.startswith("step_") and os.path.isdir(os.path.join(self.root, name)):
                try:
                    out.append(int(name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return sorted(out)


def _rmtree_quiet(path: str) -> None:
    import shutil
    try:
        shutil.rmtree(path)
    except Exception:  # noqa: BLE001
        pass


def resolve_checkpoint_path(path: str) -> str:
    """Resolve a resume target: a step dir / state.pt as-is, or a root dir -> its latest.

    Lets ``--resume experiments/run/checkpoints`` follow the latest.json pointer, while an
    explicit ``--resume .../step_000200`` loads exactly that immutable checkpoint.
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "state.pt")):
            return path
        # A root directory with a latest.json pointer.
        mgr = CheckpointManager(path)
        latest = mgr.latest_path()
        if latest:
            return latest
    return path
