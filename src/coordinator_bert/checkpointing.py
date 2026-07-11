"""Checkpoint save / resume with full training-state restoration.

A checkpoint captures everything needed to resume a run bit-for-bit (modulo backend
nondeterminism): model weights, optimizer state, LR-scheduler state, AMP scaler state, the
global step, the config dict, and RNG state for Python/NumPy/Torch (CPU + CUDA).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch


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
            # Different device count on resume; skip CUDA RNG restore.
            pass


def _as_byte_tensor(t: Any) -> torch.Tensor:
    # torch RNG state must be a ByteTensor on CPU.
    if isinstance(t, torch.Tensor):
        return t.cpu().to(torch.uint8)
    return torch.tensor(t, dtype=torch.uint8)


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
) -> str:
    """Save a checkpoint directory containing state.pt and a human-readable meta.json."""
    os.makedirs(path, exist_ok=True)

    payload: dict[str, Any] = {
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

    torch.save(payload, os.path.join(path, "state.pt"))

    # Small readable sidecar (no tensors) for quick inspection.
    meta = {
        "global_step": global_step,
        "config": _config_to_dict(config),
        "metadata": metadata or {},
    }
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return path


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
) -> dict[str, Any]:
    """Load a checkpoint and restore in-place. Returns the raw payload."""
    state_file = os.path.join(path, "state.pt") if os.path.isdir(path) else path
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


def _config_to_dict(config: Any) -> Any:
    if config is None:
        return None
    try:
        return asdict(config)
    except TypeError:
        return config
