#!/usr/bin/env python3
"""Environment diagnostics for bert_cord — human-readable + JSON, with gating modes.

Reports OS/arch/Python/PyTorch/Git, the project-selected device, MPS/CUDA/BF16/cuDNN/SDPA
capabilities, GPU name/compute-capability/memory, disk space, a filesystem write test, key
package versions, and the resolved project runtime for a given config.

Modes:
    python scripts/check_environment.py
    python scripts/check_environment.py --json environment.json
    python scripts/check_environment.py --require training
    python scripts/check_environment.py --require dgx

`--require dgx` exits non-zero when CUDA/GPU is absent, BF16 is missing (when required by the
profile), the project dir is not writable, or critical training packages are missing. It does
NOT fail on a Mac merely because CUDA is absent unless `--require dgx` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Packages considered critical for the full training loop.
_TRAINING_PACKAGES = ["accelerate", "datasets", "tokenizers", "safetensors"]
_REPORTED_PACKAGES = ["torch", "numpy", "pyyaml", "accelerate", "datasets", "tokenizers",
                      "safetensors", "psutil", "matplotlib", "scipy", "wandb"]


def _pkg_version(name: str):
    try:
        from importlib.metadata import PackageNotFoundError, version
        # pyyaml distributes as "PyYAML"
        dist = {"pyyaml": "PyYAML"}.get(name, name)
        try:
            return version(dist)
        except PackageNotFoundError:
            return None
    except Exception:  # noqa: BLE001
        return None


def _git(*args) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             cwd=_PROJECT_ROOT)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _filesystem_write_test(path: str) -> dict:
    """Verify we can create + read a file under ``path``. Writability is judged by the write
    itself; failure to unlink afterwards (some synced/network mounts disallow it) is reported
    but does not count as non-writable."""
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(path, "envcheck_write_test.tmp")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("bert_cord write test")
        with open(probe, "r", encoding="utf-8") as fh:
            ok = fh.read() == "bert_cord write test"
    except Exception as e:  # noqa: BLE001
        return {"writable": False, "error": str(e), "cleanup_ok": False}
    cleanup_ok = True
    try:
        os.remove(probe)
    except Exception:  # noqa: BLE001
        cleanup_ok = False
    return {"writable": bool(ok), "error": None if ok else "readback mismatch",
            "cleanup_ok": cleanup_ok}


def gather(config_path: str | None) -> dict:
    import torch  # local import so --help works without torch quirks

    from coordinator_bert import runtime as rt

    feats = rt.detect_features()

    # Resolve the project's selected device/runtime for the given (or default) config.
    resolved = None
    config_used = None
    req_device = "auto"
    req_precision = "auto"
    try:
        from coordinator_bert.configuration import load_config
        if config_path and os.path.exists(config_path):
            cfg = load_config(config_path)
            config_used = config_path
            req_device = cfg.runtime.device
            req_precision = cfg.train.precision
            resolved = rt.resolve_runtime(cfg.runtime, cfg.train.precision, feats)
        else:
            from coordinator_bert.configuration import RuntimeConfig
            resolved = rt.resolve_runtime(RuntimeConfig(), "auto", feats)
    except Exception as e:  # noqa: BLE001
        resolved = None
        config_used = f"(config load failed: {e})"

    total, used, free = shutil.disk_usage(_PROJECT_ROOT)
    report = {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": _git("rev-parse", "--short", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain")),
        "project_selected_device": (resolved.device if resolved else None),
        "requested_device": req_device,
        "requested_precision": req_precision,
        "resolved_precision": (resolved.precision if resolved else None),
        "mps_built": feats["mps_built"],
        "mps_available": feats["mps_available"],
        "cuda_available": feats["cuda_available"],
        "cuda_build_version": feats["cuda_build_version"],
        "cudnn_version": feats["cudnn_version"],
        "gpu_name": feats["gpu_name"],
        "compute_capability": feats["compute_capability"],
        "gpu_mem_total_bytes": feats["gpu_mem_total_bytes"],
        "gpu_mem_free_bytes": feats["gpu_mem_free_bytes"],
        "bf16_supported": feats["bf16_supported"],
        "tf32_capable": feats["tf32_capable"],
        "sdpa_available": feats["sdpa_available"],
        "flash_sdp_available": feats["flash_sdp_available"],
        "mem_efficient_sdp_available": feats["mem_efficient_sdp_available"],
        "fused_adamw_available": feats["fused_adamw_available"],
        "torch_compile_available": feats["torch_compile_available"],
        "disk_free_bytes": int(free),
        "disk_total_bytes": int(total),
        "filesystem_write_test": _filesystem_write_test(
            os.path.join(_PROJECT_ROOT, "experiments")),
        "packages": {p: _pkg_version(p) for p in _REPORTED_PACKAGES},
        "config_used": config_used,
        "resolved_runtime": (resolved.to_dict() if resolved else None),
    }
    return report


def _fmt_bytes(n) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def print_human(r: dict) -> None:
    print("=" * 72)
    print("bert_cord environment check")
    print("-" * 72)
    print(f"OS / arch          : {r['os']} {r['os_release']} / {r['arch']}")
    print(f"Python             : {r['python_version']}")
    print(f"PyTorch            : {r['torch_version']}")
    print(f"Git commit         : {r['git_commit']}  (working tree "
          f"{'DIRTY' if r['git_dirty'] else 'clean'})")
    print(f"Project device     : {r['project_selected_device']} "
          f"(requested {r['requested_device']})")
    print(f"Precision          : requested {r['requested_precision']} -> "
          f"resolved {r['resolved_precision']}")
    print(f"MPS                 : built={r['mps_built']} available={r['mps_available']}")
    print(f"CUDA                : available={r['cuda_available']} "
          f"build={r['cuda_build_version']} cudnn={r['cudnn_version']}")
    if r["cuda_available"]:
        print(f"GPU                : {r['gpu_name']} (cc {r['compute_capability']})")
        print(f"GPU memory         : free {_fmt_bytes(r['gpu_mem_free_bytes'])} / "
              f"total {_fmt_bytes(r['gpu_mem_total_bytes'])}")
    print(f"BF16 supported     : {r['bf16_supported']}   TF32 capable: {r['tf32_capable']}")
    print(f"SDPA available     : {r['sdpa_available']} "
          f"(flash={r['flash_sdp_available']}, mem_eff={r['mem_efficient_sdp_available']})")
    print(f"Fused AdamW / compile : {r['fused_adamw_available']} / "
          f"{r['torch_compile_available']}")
    print(f"Disk free          : {_fmt_bytes(r['disk_free_bytes'])} / "
          f"{_fmt_bytes(r['disk_total_bytes'])}")
    wt = r["filesystem_write_test"]
    print(f"FS write test      : {'OK' if wt['writable'] else 'FAILED: ' + str(wt['error'])}")
    pkgs = r["packages"]
    print("Packages           : " + ", ".join(
        f"{k}={pkgs[k]}" for k in ["accelerate", "datasets", "tokenizers", "matplotlib"]))
    print(f"                     scipy={pkgs.get('scipy')} (optional), "
          f"wandb={pkgs.get('wandb')} (optional)")
    print(f"Config used        : {r['config_used']}")
    print("=" * 72)


def evaluate_requirements(r: dict, require: str, require_bf16: bool) -> tuple[int, list]:
    """Return (exit_code, failures). training/dgx gate on packages, writability, hardware."""
    failures: list = []
    if require in ("training", "dgx"):
        missing = [p for p in _TRAINING_PACKAGES if r["packages"].get(p) is None]
        if missing:
            failures.append(f"missing training packages: {', '.join(missing)}")
        if not r["filesystem_write_test"]["writable"]:
            failures.append("project directory is not writable")
    if require == "dgx":
        if not r["cuda_available"]:
            failures.append("CUDA is not available")
        elif r["gpu_name"] is None:
            failures.append("no CUDA GPU is visible")
        if require_bf16 and not r["bf16_supported"]:
            failures.append("BF16 is unavailable (required by --require dgx --require-bf16)")
    return (1 if failures else 0), failures


def main() -> int:
    p = argparse.ArgumentParser(description="bert_cord environment diagnostics.")
    p.add_argument("--config", default="configs/bert_25m.yaml",
                   help="Config whose resolved runtime is reported (default bert_25m.yaml).")
    p.add_argument("--json", default=None, help="Also write the full report to this JSON path.")
    p.add_argument("--require", choices=["none", "training", "dgx"], default="none",
                   help="Exit non-zero if requirements for this profile are unmet.")
    p.add_argument("--require-bf16", action="store_true",
                   help="With --require dgx, also require BF16 support.")
    args = p.parse_args()

    report = gather(args.config)
    print_human(report)

    exit_code, failures = evaluate_requirements(report, args.require, args.require_bf16)
    report["require_profile"] = args.require
    report["requirement_failures"] = failures
    report["ok"] = exit_code == 0

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"[env] wrote JSON report -> {args.json}")

    if args.require != "none":
        if failures:
            print(f"[env] --require {args.require} FAILED:")
            for f in failures:
                print(f"   - {f}")
        else:
            print(f"[env] --require {args.require}: all requirements satisfied.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
