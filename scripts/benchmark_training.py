#!/usr/bin/env python3
"""Short, controlled, NON-scientific training benchmark (run before long training).

Measures device/precision performance with a lightweight manual loop (no Accelerate, so the
per-phase timing is clean). Warmup steps are excluded from all rates. This is a *performance*
probe, not a science run — it does not claim convergence and does not modify training configs.

Example:
  python scripts/benchmark_training.py \
    --config configs/bert_25m_dgx_portability.yaml \
    --steps 50 \
    --output-dir experiments/benchmarks/dgx_portability

Outputs (under --output-dir): environment.json, resolved_config.yaml, metrics.jsonl,
benchmark_summary.json, benchmark_report.md, plots/.

An OPTIONAL bounded batch-size probe (--probe-batch-sizes) tries {8,16,32,64,128} (or a custom
bounded list), stops safely on OOM, clears the CUDA cache, and reports the largest that fit.
It never modifies the training config and must be requested explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from coordinator_bert import runtime as rt  # noqa: E402
from coordinator_bert.checkpointing import CheckpointManager  # noqa: E402
from coordinator_bert.configuration import load_config  # noqa: E402
from coordinator_bert.data import build_dataloaders  # noqa: E402
from coordinator_bert.model import BertForMaskedLM, count_parameters  # noqa: E402

from pretrain_mlm import build_optimizer, build_scheduler, evaluate as _mlm_eval_unused  # noqa: E402,F401


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:  # noqa: BLE001
            pass


def _peak_ram_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:  # noqa: BLE001
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return rss / 1024 if sys.platform != "darwin" else rss / (1024 ** 2)
        except Exception:  # noqa: BLE001
            return float("nan")


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _make_grad_scaler(enabled: bool):
    """GradScaler (fp16 only) using the non-deprecated API when available."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # noqa: BLE001
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_ctx(resolved):
    if resolved.use_amp and resolved.device in ("cuda", "cpu"):
        return torch.autocast(device_type=resolved.device, dtype=resolved.amp_dtype)
    return nullcontext()


def _move(batch, device, non_blocking):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def run_benchmark(cfg, resolved, steps: int, warmup: int, do_eval: bool,
                  do_checkpoint: bool, seed: int, out_dir: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolved.device
    rt.apply_backend_flags(resolved)

    model = BertForMaskedLM(cfg.model).to(device)
    model = rt.maybe_compile(model, resolved)
    model.train()
    optimizer = build_optimizer(model, cfg.train, extra_adamw_kwargs=rt.adamw_extra_kwargs(resolved))
    scheduler = build_scheduler(optimizer, cfg.train)
    scaler = _make_grad_scaler(resolved.precision == "fp16")

    train_loader, val_loader, _ = build_dataloaders(cfg.model, cfg.train, cfg.data,
                                                    runtime=resolved)
    data_iter = _cycle(train_loader)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    bs = cfg.train.per_device_batch_size
    seq = cfg.train.max_seq_length
    per_step = []  # dicts of phase timings for measured (post-warmup) steps
    metrics_rows = []
    total_tokens = 0

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        batch = next(data_iter)
        batch = _move(batch, device, resolved.non_blocking)
        _sync(device)
        t_data = time.perf_counter() - t0

        t1 = time.perf_counter()
        with _autocast_ctx(resolved):
            out = model(**batch)
            loss = out["loss"]
        _sync(device)
        t_fwd = time.perf_counter() - t1

        t2 = time.perf_counter()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        _sync(device)
        t_bwd = time.perf_counter() - t2

        t3 = time.perf_counter()
        if cfg.train.max_grad_norm > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        t_opt = time.perf_counter() - t3

        step_tokens = int(batch["attention_mask"].sum())
        total_tokens += step_tokens
        step_latency = t_data + t_fwd + t_bwd + t_opt
        row = {"step": step, "warmup": step <= warmup, "latency_s": step_latency,
               "data_s": t_data, "fwd_s": t_fwd, "bwd_s": t_bwd, "opt_s": t_opt,
               "tokens": step_tokens, "loss": float(loss.detach())}
        metrics_rows.append(row)
        if step > warmup:
            per_step.append(row)

    # Optional eval + checkpoint overheads (single measured occurrence).
    eval_overhead = None
    if do_eval:
        _sync(device)
        te = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for i, b in enumerate(val_loader):
                if i >= cfg.train.eval_max_batches:
                    break
                b = _move(b, device, resolved.non_blocking)
                with _autocast_ctx(resolved):
                    model(**b)
        _sync(device)
        model.train()
        eval_overhead = time.perf_counter() - te

    checkpoint_overhead = None
    if do_checkpoint:
        mgr = CheckpointManager(os.path.join(out_dir, "checkpoints"))
        _sync(device)
        tc = time.perf_counter()
        mgr.save(steps, model=model, optimizer=optimizer, scheduler=scheduler,
                 config=cfg.model, precision=resolved.precision, device=device)
        checkpoint_overhead = time.perf_counter() - tc

    # Aggregate (warmup excluded).
    lat = [r["latency_s"] for r in per_step]
    measured_tokens = sum(r["tokens"] for r in per_step)
    measured_time = sum(lat)
    n = max(1, len(per_step))
    summary = {
        "kind": "training_benchmark",
        "note": "performance probe only; NOT a scientific/convergence result",
        "device": device,
        "precision": resolved.precision,
        "steps_total": steps,
        "warmup_steps": warmup,
        "measured_steps": len(per_step),
        "effective_batch_size": bs * cfg.train.gradient_accumulation_steps,
        "per_device_batch_size": bs,
        "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
        "sequence_length": seq,
        "steps_per_s": (len(per_step) / measured_time) if measured_time else None,
        "tokens_per_s": (measured_tokens / measured_time) if measured_time else None,
        "samples_per_s": ((bs * len(per_step)) / measured_time) if measured_time else None,
        "latency_median_ms": (stats.median(lat) * 1e3) if lat else None,
        "latency_p90_ms": (float(np.percentile(lat, 90)) * 1e3) if lat else None,
        "latency_p99_ms": (float(np.percentile(lat, 99)) * 1e3) if lat else None,
        "forward_mean_ms": _mean_ms(per_step, "fwd_s"),
        "backward_mean_ms": _mean_ms(per_step, "bwd_s"),
        "optimizer_mean_ms": _mean_ms(per_step, "opt_s"),
        "dataloader_wait_mean_ms": _mean_ms(per_step, "data_s"),
        "eval_overhead_s": eval_overhead,
        "checkpoint_overhead_s": checkpoint_overhead,
        "peak_ram_mb": _peak_ram_mb(),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / (1024 ** 2)
                         if device == "cuda" else None),
        "total_tokens_processed": total_tokens,
        "param_count": count_parameters(model)["unique"],
        "runtime_notes": resolved.notes,
    }
    return {"summary": summary, "metrics_rows": metrics_rows}


def _mean_ms(rows, key):
    vals = [r[key] for r in rows]
    return (stats.mean(vals) * 1e3) if vals else None


def batch_probe(cfg, resolved, candidates, seq_override=None) -> dict:
    """Bounded, OOM-safe batch-size probe. Never modifies the training config."""
    device = resolved.device
    seq = seq_override or cfg.train.max_seq_length
    vocab = cfg.model.vocab_size
    results = []
    largest_ok = None
    model = BertForMaskedLM(cfg.model).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for bs in candidates:
        ok = False
        err = None
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(5, vocab, (bs, seq), device=device)
            am = torch.ones(bs, seq, dtype=torch.long, device=device)
            lab = torch.randint(5, vocab, (bs, seq), device=device)
            with _autocast_ctx(resolved):
                out = model(input_ids=ids, attention_mask=am, labels=lab)
            out["loss"].backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            ok = True
            largest_ok = bs
        except RuntimeError as e:  # noqa: BLE001
            err = str(e).splitlines()[0]
            if "out of memory" in str(e).lower() and device == "cuda":
                torch.cuda.empty_cache()
            # stop escalating on OOM
            results.append({"batch_size": bs, "ok": False, "error": err})
            break
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()
        results.append({"batch_size": bs, "ok": ok,
                        "peak_vram_mb": (torch.cuda.max_memory_allocated() / (1024 ** 2)
                                         if device == "cuda" and ok else None)})
    return {"candidates": candidates, "sequence_length": seq, "largest_ok": largest_ok,
            "results": results,
            "note": "diagnostic only; does NOT modify the training config"}


# --------------------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------------------- #
def _write_environment(out_dir: str) -> None:
    feats = rt.detect_features()
    import platform
    env = {"os": platform.system(), "arch": platform.machine(),
           "python": platform.python_version(), "features": feats}
    with open(os.path.join(out_dir, "environment.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, default=str)


def _write_resolved_config(cfg, out_dir: str) -> None:
    import yaml
    with open(os.path.join(out_dir, "resolved_config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)


def _write_metrics(rows, out_dir: str) -> None:
    with open(os.path.join(out_dir, "metrics.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_plots(rows, summary, out_dir: str) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    paths = []

    steps = [r["step"] for r in rows]
    lat = [r["latency_s"] * 1e3 for r in rows]
    warm = [r["warmup"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, lat, marker="o", ms=3, color="#1f77b4", label="step latency (ms)")
    warm_steps = [s for s, w in zip(steps, warm) if w]
    if warm_steps:
        ax.axvspan(min(warm_steps), max(warm_steps), color="#cccccc", alpha=0.4,
                   label="warmup (excluded)")
    ax.set_xlabel("step"); ax.set_ylabel("latency (ms)")
    ax.set_title(f"Step latency — {summary['device']}/{summary['precision']}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    p = os.path.join(plots_dir, "step_latency.png"); fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig); paths.append(p)

    phases = ["forward_mean_ms", "backward_mean_ms", "optimizer_mean_ms",
              "dataloader_wait_mean_ms"]
    vals = [summary.get(k) or 0.0 for k in phases]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([p.replace("_mean_ms", "") for p in phases], vals, color="#4c72b0")
    ax.set_ylabel("mean time per step (ms)")
    ax.set_title("Per-phase time breakdown (warmup excluded)")
    ax.grid(True, axis="y", alpha=0.25)
    p = os.path.join(plots_dir, "phase_breakdown.png"); fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig); paths.append(p)
    return paths


def _write_report(summary, plot_paths, probe, out_dir: str) -> None:
    def f(x, unit=""):
        return "n/a" if x is None else (f"{x:.3f}{unit}" if isinstance(x, float) else f"{x}{unit}")
    lines = [f"# Training benchmark — {summary['device']} / {summary['precision']}", "",
             "> Performance probe only. **Not** a scientific/convergence result; the config is "
             "not modified. Warmup steps are excluded from all rates.", "",
             "## Throughput", "",
             f"- Steps/s: **{f(summary['steps_per_s'])}**",
             f"- Tokens/s: **{f(summary['tokens_per_s'])}**",
             f"- Samples/s: {f(summary['samples_per_s'])}",
             f"- Measured steps: {summary['measured_steps']} (warmup {summary['warmup_steps']})",
             "", "## Latency", "",
             f"- Median: {f(summary['latency_median_ms'],' ms')} | "
             f"p90: {f(summary['latency_p90_ms'],' ms')} | p99: {f(summary['latency_p99_ms'],' ms')}",
             f"- Forward: {f(summary['forward_mean_ms'],' ms')} | "
             f"Backward: {f(summary['backward_mean_ms'],' ms')} | "
             f"Optimizer: {f(summary['optimizer_mean_ms'],' ms')} | "
             f"Dataloader wait: {f(summary['dataloader_wait_mean_ms'],' ms')}",
             "", "## Resources & config", "",
             f"- Peak RAM: {f(summary['peak_ram_mb'],' MB')} | "
             f"Peak VRAM: {f(summary['peak_vram_mb'],' MB')}",
             f"- Effective batch: {summary['effective_batch_size']} "
             f"(per-device {summary['per_device_batch_size']} × "
             f"accum {summary['gradient_accumulation_steps']}) | "
             f"seq len {summary['sequence_length']}",
             f"- Total tokens processed: {summary['total_tokens_processed']:,}",
             f"- Eval overhead: {f(summary['eval_overhead_s'],' s')} | "
             f"Checkpoint overhead: {f(summary['checkpoint_overhead_s'],' s')}",
             f"- Params: {summary['param_count']:,}", ""]
    if summary.get("runtime_notes"):
        lines += ["## Runtime notes", ""] + [f"- {n}" for n in summary["runtime_notes"]] + [""]
    if probe:
        lines += ["## Batch-size probe (diagnostic; config unchanged)", "",
                  f"- Candidates: {probe['candidates']} @ seq {probe['sequence_length']}",
                  f"- Largest that fit: **{probe['largest_ok']}**", ""]
    if plot_paths:
        lines += ["## Plots", ""]
        for p in plot_paths:
            lines.append(f"![{os.path.basename(p)}](plots/{os.path.basename(p)})")
        lines.append("")
    with open(os.path.join(out_dir, "benchmark_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Short controlled training benchmark.")
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", action="store_true", help="Measure one evaluation pass overhead.")
    p.add_argument("--checkpoint", action="store_true",
                   help="Measure one checkpoint save overhead.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--probe-batch-sizes", action="store_true",
                   help="Run the bounded OOM-safe batch-size probe (explicit opt-in).")
    p.add_argument("--probe-candidates", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    args = p.parse_args()

    if args.steps > 200:
        print(f"[benchmark] refusing --steps {args.steps} > 200 (conservative cap; raise "
              "explicitly in code if truly needed).", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    resolved = rt.resolve_runtime(cfg.runtime, cfg.train.precision)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"benchmark :: {args.config}")
    for line in rt.runtime_report_lines(resolved):
        print("  " + line)
    print("=" * 70)

    _write_environment(args.output_dir)
    _write_resolved_config(cfg, args.output_dir)

    result = run_benchmark(cfg, resolved, args.steps, args.warmup, args.eval,
                           args.checkpoint, args.seed, args.output_dir)
    summary = result["summary"]
    _write_metrics(result["metrics_rows"], args.output_dir)

    probe = None
    if args.probe_batch_sizes:
        print("[benchmark] running bounded batch-size probe (config NOT modified)...")
        probe = batch_probe(cfg, resolved, args.probe_candidates)
        summary["batch_probe"] = probe
        print(f"[benchmark] largest batch that fit: {probe['largest_ok']}")

    plot_paths = [] if args.no_plots else _make_plots(result["metrics_rows"], summary,
                                                      args.output_dir)
    with open(os.path.join(args.output_dir, "benchmark_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _write_report(summary, plot_paths, probe, args.output_dir)

    print(f"[benchmark] steps/s={_fmt(summary['steps_per_s'])} "
          f"tokens/s={_fmt(summary['tokens_per_s'])} "
          f"median_latency={_fmt(summary['latency_median_ms'])}ms")
    print(f"[benchmark] outputs -> {args.output_dir}")
    return 0


def _fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
