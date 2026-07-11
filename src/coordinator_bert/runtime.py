"""Platform-aware runtime: device/precision resolution and optional performance features.

Scientific settings (model, seed, LR, batch, sequence length, optimizer, scheduler) live in
the model/train config. This module owns the *hardware/runtime* concerns and resolves them
against the machine actually present, with safe fallbacks:

  * device selection (auto -> cuda | mps | cpu),
  * precision (bf16 only when CUDA truly supports it; mps/cpu -> fp32),
  * TF32, SDPA, pinned memory, persistent workers, non-blocking copies, fused AdamW,
    torch.compile — all **feature-detected, optional, reported, and safely disabled** when
    unavailable.

Nothing here enables torch.compile by default and nothing imports CUDA-only or FlashAttention
packages. Everything degrades to a correct CPU/fp32 path.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from .configuration import RuntimeConfig  # config schema lives in configuration.py

__all__ = [
    "RuntimeConfig", "ResolvedRuntime", "detect_features", "resolve_device",
    "resolve_precision", "resolve_runtime", "apply_backend_flags", "maybe_compile",
    "adamw_extra_kwargs", "runtime_report_lines",
]


# --------------------------------------------------------------------------------------- #
# Feature detection (no side effects; never raises)
# --------------------------------------------------------------------------------------- #
def detect_features() -> dict:
    """Best-effort capability probe. Every field is safe to read on any platform."""
    feats: dict = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_build_version": torch.version.cuda,
        "cudnn_version": None,
        "gpu_name": None,
        "compute_capability": None,
        "gpu_mem_total_bytes": None,
        "gpu_mem_free_bytes": None,
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()),
        "bf16_supported": False,
        "tf32_capable": False,
        "sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        "flash_sdp_available": None,
        "mem_efficient_sdp_available": None,
        "fused_adamw_available": _fused_adamw_available(),
        "torch_compile_available": hasattr(torch, "compile"),
    }
    if feats["cuda_available"]:
        try:
            feats["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            pass
        try:
            idx = torch.cuda.current_device()
            feats["gpu_name"] = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            feats["compute_capability"] = f"{cap[0]}.{cap[1]}"
            feats["tf32_capable"] = cap[0] >= 8  # Ampere+
            free, total = torch.cuda.mem_get_info(idx)
            feats["gpu_mem_free_bytes"] = int(free)
            feats["gpu_mem_total_bytes"] = int(total)
        except Exception:  # noqa: BLE001
            pass
        try:
            feats["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001
            feats["bf16_supported"] = False
        # SDPA backend availability (best-effort; API varies across torch versions).
        try:
            feats["flash_sdp_available"] = bool(torch.backends.cuda.flash_sdp_enabled())
            feats["mem_efficient_sdp_available"] = bool(
                torch.backends.cuda.mem_efficient_sdp_enabled())
        except Exception:  # noqa: BLE001
            pass
    return feats


def _fused_adamw_available() -> bool:
    """Whether torch.optim.AdamW exposes the fused path (CUDA-only at runtime)."""
    try:
        import inspect
        return "fused" in inspect.signature(torch.optim.AdamW).parameters
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------------------- #
def resolve_device(requested: str, features: Optional[dict] = None) -> tuple[str, list]:
    """Resolve a requested device to a concrete one, with fallback notes."""
    feats = features or detect_features()
    requested = (requested or "auto").lower()
    notes: list = []

    if requested == "auto":
        if feats["cuda_available"]:
            return "cuda", notes
        if feats["mps_available"]:
            return "mps", notes
        return "cpu", notes
    if requested == "cuda":
        if feats["cuda_available"]:
            return "cuda", notes
        notes.append("device=cuda requested but CUDA unavailable -> cpu")
        return "cpu", notes
    if requested == "mps":
        if feats["mps_available"]:
            return "mps", notes
        notes.append("device=mps requested but MPS unavailable -> cpu")
        return "cpu", notes
    return "cpu", notes


def resolve_precision(requested: str, device: str,
                      features: Optional[dict] = None) -> tuple[str, list]:
    """Resolve requested precision to a concrete one honoring real hardware support."""
    feats = features or detect_features()
    requested = (requested or "auto").lower()
    notes: list = []
    bf16_ok = device == "cuda" and feats.get("bf16_supported", False)

    if requested == "auto":
        return ("bf16", notes) if bf16_ok else ("fp32", notes)
    if requested == "bf16":
        if bf16_ok:
            return "bf16", notes
        notes.append(f"precision=bf16 requested but unsupported on {device} -> fp32")
        return "fp32", notes
    if requested == "fp16":
        if device == "cuda":
            return "fp16", notes
        notes.append(f"precision=fp16 requested but CUDA absent ({device}) -> fp32")
        return "fp32", notes
    return "fp32", notes


_AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class ResolvedRuntime:
    device: str
    precision: str
    amp_dtype_str: str
    allow_tf32: bool
    pin_memory: bool
    num_workers: int
    persistent_workers: bool
    non_blocking: bool
    fused_adamw: bool
    torch_compile: bool
    compile_mode: str
    features: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def amp_dtype(self):
        return _AMP_DTYPE[self.precision]

    @property
    def use_amp(self) -> bool:
        return self.precision in ("bf16", "fp16")

    @property
    def torch_device(self) -> "torch.device":
        return torch.device(self.device)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def resolve_runtime(runtime_cfg: RuntimeConfig, requested_precision: str,
                    features: Optional[dict] = None) -> ResolvedRuntime:
    """Resolve all runtime settings against the present machine, disabling what is unavailable."""
    feats = features or detect_features()
    notes: list = []

    device, dnotes = resolve_device(runtime_cfg.device, feats)
    notes += dnotes
    precision, pnotes = resolve_precision(requested_precision, device, feats)
    notes += pnotes

    is_cuda = device == "cuda"

    # TF32 only meaningful on Ampere+ CUDA.
    allow_tf32 = bool(runtime_cfg.allow_tf32 and is_cuda and feats.get("tf32_capable", False))
    if runtime_cfg.allow_tf32 and not allow_tf32:
        notes.append("allow_tf32 requested but not applicable (needs Ampere+ CUDA) -> off")

    # Pinned memory / non-blocking copies are CUDA-only.
    pin_memory = bool(runtime_cfg.pin_memory and is_cuda)
    if runtime_cfg.pin_memory and not pin_memory:
        notes.append("pin_memory requested but only useful on CUDA -> off")
    non_blocking = bool(runtime_cfg.non_blocking and is_cuda)
    if runtime_cfg.non_blocking and not non_blocking:
        notes.append("non_blocking requested but only applies to CUDA copies -> off")

    # persistent_workers requires num_workers > 0.
    num_workers = max(0, int(runtime_cfg.num_workers))
    persistent_workers = bool(runtime_cfg.persistent_workers and num_workers > 0)
    if runtime_cfg.persistent_workers and not persistent_workers:
        notes.append("persistent_workers requested but num_workers==0 -> off")

    # fused AdamW: available API + CUDA.
    fused_adamw = bool(runtime_cfg.fused_adamw and is_cuda
                       and feats.get("fused_adamw_available", False))
    if runtime_cfg.fused_adamw and not fused_adamw:
        notes.append("fused_adamw requested but unavailable on this device -> standard AdamW")

    # torch.compile: opt-in only, needs the API.
    torch_compile = bool(runtime_cfg.torch_compile and feats.get("torch_compile_available", False))
    if runtime_cfg.torch_compile and not torch_compile:
        notes.append("torch_compile requested but torch.compile unavailable -> off")

    return ResolvedRuntime(
        device=device, precision=precision, amp_dtype_str=precision,
        allow_tf32=allow_tf32, pin_memory=pin_memory, num_workers=num_workers,
        persistent_workers=persistent_workers, non_blocking=non_blocking,
        fused_adamw=fused_adamw, torch_compile=torch_compile,
        compile_mode=runtime_cfg.compile_mode, features=feats, notes=notes,
    )


def apply_backend_flags(resolved: ResolvedRuntime) -> None:
    """Apply process-wide backend flags (TF32). Safe no-op off CUDA."""
    if resolved.device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = resolved.allow_tf32
            torch.backends.cudnn.allow_tf32 = resolved.allow_tf32
        except Exception:  # noqa: BLE001
            pass


def maybe_compile(model, resolved: ResolvedRuntime):
    """Compile the model only when explicitly enabled and available; else return as-is."""
    if resolved.torch_compile and hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode=resolved.compile_mode)
        except Exception as e:  # noqa: BLE001
            resolved.notes.append(f"torch.compile failed ({e}); using eager model")
    return model


def adamw_extra_kwargs(resolved: ResolvedRuntime) -> dict:
    """Extra kwargs for torch.optim.AdamW (fused=True only when resolved on)."""
    return {"fused": True} if resolved.fused_adamw else {}


def runtime_report_lines(resolved: ResolvedRuntime) -> list:
    """Human-readable lines describing the resolved runtime that materially affects training."""
    f = resolved.features
    lines = [
        f"device            : {resolved.device}",
        f"precision         : {resolved.precision}  (amp={'on' if resolved.use_amp else 'off'})",
        f"allow_tf32        : {resolved.allow_tf32}",
        f"sdpa_available    : {f.get('sdpa_available')}",
        f"pin_memory        : {resolved.pin_memory}",
        f"num_workers       : {resolved.num_workers}",
        f"persistent_workers: {resolved.persistent_workers}",
        f"non_blocking      : {resolved.non_blocking}",
        f"fused_adamw       : {resolved.fused_adamw}",
        f"torch_compile     : {resolved.torch_compile}"
        + (f" (mode={resolved.compile_mode})" if resolved.torch_compile else ""),
    ]
    if resolved.device == "cuda":
        lines.append(f"gpu               : {f.get('gpu_name')} "
                     f"(cc {f.get('compute_capability')}, cuda {f.get('cuda_build_version')}, "
                     f"cudnn {f.get('cudnn_version')}, bf16 {f.get('bf16_supported')})")
    elif resolved.device == "mps":
        lines.append("gpu               : Apple MPS (bf16/fp16 training unsupported -> fp32)")
    if resolved.notes:
        for n in resolved.notes:
            lines.append(f"note              : {n}")
    return lines
