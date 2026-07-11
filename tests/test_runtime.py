"""Feature-resolution tests for the platform-aware runtime — no real GPU required.

The resolver accepts an injected ``features`` dict, so CUDA/MPS behaviour is exercised on any
machine (including this CPU-only CI) by describing a synthetic machine.
"""

from __future__ import annotations

from coordinator_bert.configuration import RuntimeConfig
from coordinator_bert import runtime as rt


def _feats(**over):
    base = {
        "cuda_available": False, "mps_available": False, "mps_built": False,
        "bf16_supported": False, "tf32_capable": False, "sdpa_available": True,
        "fused_adamw_available": True, "torch_compile_available": True,
        "cuda_build_version": None, "cudnn_version": None, "gpu_name": None,
        "compute_capability": None,
    }
    base.update(over)
    return base


def test_resolve_device_auto_prefers_cuda_then_mps_then_cpu():
    assert rt.resolve_device("auto", _feats(cuda_available=True))[0] == "cuda"
    assert rt.resolve_device("auto", _feats(mps_available=True))[0] == "mps"
    assert rt.resolve_device("auto", _feats())[0] == "cpu"


def test_resolve_device_explicit_falls_back_with_note():
    dev, notes = rt.resolve_device("cuda", _feats())  # no CUDA present
    assert dev == "cpu" and any("cuda" in n.lower() for n in notes)
    dev, notes = rt.resolve_device("mps", _feats())
    assert dev == "cpu" and any("mps" in n.lower() for n in notes)


def test_resolve_precision_bf16_only_on_capable_cuda():
    # auto -> bf16 on bf16-capable CUDA
    assert rt.resolve_precision("auto", "cuda", _feats(cuda_available=True,
                                                       bf16_supported=True))[0] == "bf16"
    # auto -> fp32 without bf16
    assert rt.resolve_precision("auto", "cpu", _feats())[0] == "fp32"
    # explicit bf16 on cpu -> fp32 + note
    p, notes = rt.resolve_precision("bf16", "cpu", _feats())
    assert p == "fp32" and notes
    # explicit bf16 on cuda without support -> fp32 + note
    p, notes = rt.resolve_precision("bf16", "cuda", _feats(cuda_available=True,
                                                           bf16_supported=False))
    assert p == "fp32" and notes


def test_resolve_precision_fp16_requires_cuda():
    assert rt.resolve_precision("fp16", "cuda", _feats(cuda_available=True))[0] == "fp16"
    assert rt.resolve_precision("fp16", "cpu", _feats())[0] == "fp32"


def test_resolve_runtime_cpu_disables_all_perf_features():
    cfg = RuntimeConfig(device="auto", allow_tf32=True, pin_memory=True, num_workers=4,
                        persistent_workers=True, non_blocking=True, fused_adamw=True,
                        torch_compile=True)
    r = rt.resolve_runtime(cfg, "bf16", _feats())  # CPU machine
    assert r.device == "cpu" and r.precision == "fp32"
    assert not r.allow_tf32 and not r.pin_memory and not r.non_blocking and not r.fused_adamw
    # torch_compile is device-agnostic; it can stay on if the API exists.
    assert r.torch_compile is True
    # num_workers preserved but persistent still allowed only because workers>0.
    assert r.num_workers == 4 and r.persistent_workers is True
    assert r.notes  # explains the disables


def test_resolve_runtime_dgx_throughput_enables_features_on_ampere():
    feats = _feats(cuda_available=True, bf16_supported=True, tf32_capable=True,
                   compute_capability="9.0", gpu_name="DGX Spark GPU")
    cfg = RuntimeConfig(device="auto", allow_tf32=True, pin_memory=True, num_workers=4,
                        persistent_workers=True, non_blocking=True, fused_adamw=True)
    r = rt.resolve_runtime(cfg, "bf16", feats)
    assert r.device == "cuda" and r.precision == "bf16"
    assert r.allow_tf32 and r.pin_memory and r.non_blocking and r.fused_adamw
    assert r.persistent_workers and r.num_workers == 4
    assert r.use_amp is True and str(r.amp_dtype) == "torch.bfloat16"


def test_persistent_workers_requires_workers():
    feats = _feats(cuda_available=True, bf16_supported=True, tf32_capable=True)
    cfg = RuntimeConfig(device="cuda", num_workers=0, persistent_workers=True)
    r = rt.resolve_runtime(cfg, "bf16", feats)
    assert r.persistent_workers is False and any("persistent" in n for n in r.notes)


def test_fused_adamw_disabled_when_unavailable():
    feats = _feats(cuda_available=True, bf16_supported=True, fused_adamw_available=False)
    cfg = RuntimeConfig(device="cuda", fused_adamw=True)
    r = rt.resolve_runtime(cfg, "bf16", feats)
    assert r.fused_adamw is False and any("fused" in n for n in r.notes)


def test_tf32_only_on_ampere_plus():
    feats = _feats(cuda_available=True, tf32_capable=False)  # pre-Ampere
    cfg = RuntimeConfig(device="cuda", allow_tf32=True)
    r = rt.resolve_runtime(cfg, "fp32", feats)
    assert r.allow_tf32 is False and any("tf32" in n.lower() for n in r.notes)


def test_runtime_config_from_dict_ignores_unknown_keys():
    r = RuntimeConfig.from_dict({"device": "cuda", "bogus": 1, "pin_memory": True})
    assert r.device == "cuda" and r.pin_memory is True


def test_detect_features_runs_and_reports_core_fields():
    f = rt.detect_features()
    for key in ("torch_version", "cuda_available", "sdpa_available", "mps_built",
                "torch_compile_available"):
        assert key in f
    assert isinstance(f["cuda_available"], bool)


def test_report_lines_are_strings():
    r = rt.resolve_runtime(RuntimeConfig(), "auto")
    lines = rt.runtime_report_lines(r)
    assert lines and all(isinstance(x, str) for x in lines)
