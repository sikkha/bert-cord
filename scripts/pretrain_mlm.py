#!/usr/bin/env python3
"""Masked-language-model pretraining entrypoint for the custom coordinator BERT.

Features (Milestone 0):
  * YAML-driven config, deterministic seeding, parameter-count report,
  * Hugging Face Accelerate for device placement, mixed precision, and grad accumulation,
  * AdamW with decoupled weight decay (no decay on bias/LayerNorm),
  * warmup + cosine/linear decay LR schedule,
  * BF16 when CUDA reports support, safe fp32 fallback otherwise,
  * validation MLM loss + masked-token accuracy,
  * checkpoint save and resume (model / optimizer / scheduler / step / RNG).

Usage:
  python scripts/pretrain_mlm.py --config configs/bert_25m.yaml --smoke
  python scripts/pretrain_mlm.py --config configs/bert_25m.yaml --smoke \
      --resume experiments/smoke/checkpoints/last
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import random
import sys
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import torch

# Make the src package importable when run as a plain script.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.checkpointing import (  # noqa: E402
    CheckpointError,
    CheckpointManager,
    load_checkpoint,
    resolve_checkpoint_path,
)
from coordinator_bert.configuration import RunConfig, load_config  # noqa: E402
from coordinator_bert.data import build_dataloaders  # noqa: E402
from coordinator_bert.model import BertForMaskedLM, count_parameters, parameter_count_report  # noqa: E402
from coordinator_bert import runtime as rt  # noqa: E402
from coordinator_bert import tracking as tk  # noqa: E402


# --------------------------------------------------------------------------------------- #
# Precision / device helpers
# --------------------------------------------------------------------------------------- #
def resolve_precision(requested: str) -> str:
    """Resolve 'auto' to a concrete precision, honoring real hardware support.

    Returns one of {"bf16", "fp16", "fp32"}. BF16/FP16 are only chosen when CUDA actually
    supports them; otherwise we fall back to fp32 (no unsupported-precision claims).
    """
    requested = (requested or "auto").lower()
    cuda = torch.cuda.is_available()
    bf16_ok = cuda and torch.cuda.is_bf16_supported()

    if requested == "auto":
        return "bf16" if bf16_ok else "fp32"
    if requested == "bf16":
        if bf16_ok:
            return "bf16"
        print("[precision] bf16 requested but not supported on this device -> fp32 fallback")
        return "fp32"
    if requested == "fp16":
        if cuda:
            return "fp16"
        print("[precision] fp16 requested but CUDA unavailable -> fp32 fallback")
        return "fp32"
    return "fp32"


def precision_to_accelerate(prec: str) -> str:
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "no"}[prec]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def startup_report(cfg: RunConfig, precision: str, device: str, seed: int,
                   param_report: str) -> None:
    git = _git_commit()
    print("=" * 74)
    print("coordinator_bert :: MLM pretraining startup report")
    print("-" * 74)
    print(f"OS/arch          : {platform.system()} {platform.release()} / "
          f"{platform.machine()}")
    print(f"Python           : {platform.python_version()}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"CUDA (torch)     : {torch.version.cuda}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device      : {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported   : {torch.cuda.is_bf16_supported()}")
    else:
        print("CUDA device      : none (CPU)")
        print("BF16 supported   : False (no CUDA)")
    print(f"Resolved device  : {device}")
    print(f"Resolved precision: {precision}")
    print(f"Seed             : {seed}")
    print(f"Git commit       : {git}")
    print(f"{param_report}")
    print("=" * 74)


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(_SRC),
        )
        return out.stdout.strip() or "none"
    except Exception:  # noqa: BLE001
        return "unavailable"


def _git_dirty() -> bool:
    try:
        import subprocess
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                             cwd=os.path.dirname(_SRC))
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _tracking_context(cfg: RunConfig, resolved, param_count: int, device_str: str) -> dict:
    """Assemble the run-identity + config fields logged to the tracker (no secrets)."""
    feats = resolved.features or {}
    eff_batch = cfg.train.per_device_batch_size * cfg.train.gradient_accumulation_steps
    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "resolved_config": cfg.to_dict(),
        "model_architecture": cfg.model.__dict__ if hasattr(cfg.model, "__dict__")
        else str(cfg.model),
        "param_count": int(param_count),
        "seed": cfg.train.seed,
        "dataset_identity": cfg.data.dataset_name or "synthetic_copy_motif",
        "tokenizer_identity": cfg.data.tokenizer_path or "synthetic_reserved_specials",
        "per_device_batch_size": cfg.train.per_device_batch_size,
        "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
        "effective_batch_size": eff_batch,
        "sequence_length": cfg.train.max_seq_length,
        "optimizer": "AdamW" + ("(fused)" if resolved.fused_adamw else ""),
        "scheduler": cfg.train.lr_scheduler,
        "precision": resolved.precision,
        "device": device_str,
        "pytorch_version": torch.__version__,
        "cuda_version": feats.get("cuda_build_version"),
        "gpu_name": feats.get("gpu_name"),
        "bf16_supported": feats.get("bf16_supported"),
        "sdpa": cfg.model.use_sdpa,
        "checkpoint_policy": "immutable step_XXXXXX dirs + latest.json (SHA-256 verified)",
    }


def _run_name_for(cfg: RunConfig, resolved, param_count: int) -> str:
    if cfg.tracking.run_name:
        return cfg.tracking.run_name
    model_tag = f"bert{int(round(param_count / 1e6))}m"
    platform_tag = resolved.device
    experiment_tag = os.path.basename(cfg.output.dir.rstrip("/")) or "run"
    return tk.make_run_name(model_tag, platform_tag, experiment_tag)


# --------------------------------------------------------------------------------------- #
# Optimizer / scheduler
# --------------------------------------------------------------------------------------- #
def build_optimizer(model: torch.nn.Module, cfg,
                    extra_adamw_kwargs: Optional[dict] = None) -> torch.optim.Optimizer:
    """AdamW with no weight decay on bias / LayerNorm parameters.

    ``extra_adamw_kwargs`` may carry e.g. ``{"fused": True}`` when the resolved runtime enabled
    fused AdamW (CUDA only). It defaults to nothing, so behaviour is unchanged elsewhere.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "LayerNorm" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_epsilon,
        **(extra_adamw_kwargs or {}),
    )


def _write_run_artifacts(cfg: RunConfig, resolved, device_str: str) -> None:
    """Write resolved_config.yaml + environment.json into the run output dir (best-effort)."""
    import json as _json

    import yaml as _yaml
    try:
        out_dir = cfg.output.dir
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "resolved_config.yaml"), "w", encoding="utf-8") as fh:
            _yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
        env = {
            "device": device_str,
            "precision": resolved.precision,
            "runtime": resolved.to_dict(),
            "git_commit": _git_commit(),
            "param_estimate": cfg.model.estimate_num_parameters(),
        }
        with open(os.path.join(out_dir, "environment.json"), "w", encoding="utf-8") as fh:
            _json.dump(env, fh, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not write run artifacts: {e}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(0, cfg.warmup_steps)
    total = max(1, cfg.max_steps)
    min_ratio = cfg.min_lr_ratio
    kind = cfg.lr_scheduler

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        progress = min(1.0, progress)
        if kind == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine
        # linear
        return min_ratio + (1.0 - min_ratio) * (1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, dataloader, accelerator, max_batches: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    n_batches = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        outputs = model(**batch)
        total_loss += float(outputs["loss"].detach())
        logits = outputs["logits"]
        labels = batch["labels"]
        mask = labels != -100
        preds = logits.argmax(dim=-1)
        total_correct += int((preds[mask] == labels[mask]).sum())
        total_masked += int(mask.sum())
        n_batches += 1
    model.train()
    avg_loss = total_loss / max(1, n_batches)
    acc = total_correct / max(1, total_masked)
    return {
        "val_loss": avg_loss,
        "val_masked_accuracy": acc,
        "val_perplexity": math.exp(min(20.0, avg_loss)),
        "val_masked_tokens": total_masked,
    }


# --------------------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------------------- #
def apply_smoke_overrides(cfg: RunConfig) -> RunConfig:
    """Shrink the run to a fast, self-contained smoke test."""
    train = replace(
        cfg.train,
        max_steps=20,
        warmup_steps=4,
        learning_rate=5e-4,  # higher LR so a short smoke run shows a visible loss trend
        eval_every=5,
        save_every=1000,  # periodic saves off for smoke; only the final 'last' is written
        eval_max_batches=3,
        log_every=2,
        per_device_batch_size=8,
        gradient_accumulation_steps=2,
        max_seq_length=32,
    )
    syn = replace(cfg.data.synthetic, num_train_examples=256, num_val_examples=64,
                  max_len=32)
    data = replace(cfg.data, synthetic=syn, dataset_name=None)
    output = replace(
        cfg.output,
        dir="experiments/smoke",
        checkpoint_dir="experiments/smoke/checkpoints",
    )
    return replace(cfg, train=train, data=data, output=output)


def _append_metrics_row(path: str, row: dict, fmt: str) -> None:
    """Append one metrics record to a JSONL or CSV file (header written on first CSV write)."""
    import csv
    import json as _json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["step", "tokens_seen", "train_loss", "val_loss", "masked_accuracy",
              "learning_rate", "gradient_norm"]
    if fmt == "csv":
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})
    else:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({k: row.get(k) for k in fields}) + "\n")


def _early_stop_decision(records: list, es: dict):
    """Return (should_stop, status, message). Never stops on instability or thin data.

    Uses the conservative curve analyzer. 'stop' requires: PLATEAU status (which itself
    requires >= min_evals, patience/negligible-gain, and NO instability) AND a predicted gain
    over the remaining budget below the configured threshold.
    """
    from coordinator_bert.curve_analysis import AnalysisConfig, analyze

    cfg = AnalysisConfig(
        patience=es["patience"], min_delta=es["min_delta"], min_evals=es["min_evals"],
        min_fit_points=max(6, es["min_evals"]),
        future_steps=(es["future_step"],), negligible_gain_per_100=es["min_gain_per_100"],
        n_boot=120, run_id=es.get("run_id", "run"),
    )
    res = analyze(records, cfg)
    status = res.status
    # Predicted absolute gain from current best to the forecast horizon / asymptote.
    predicted_gain = float("nan")
    best = res.best_val_loss
    fut = res.predicted_val_loss.get(str(es["future_step"]))
    asy = res.predicted_asymptote.get("point")
    if best is not None:
        cand = [v for v in (fut, asy) if v is not None and v == v]
        if cand:
            predicted_gain = best - min(cand)
    gain_small = (predicted_gain != predicted_gain) or (predicted_gain < es["predicted_gain"])
    should_stop = (status == "PLATEAU") and gain_small
    msg = (f"analysis status={status}; recent Δloss/100={res.recent_improvement_per_100:.5f}; "
           f"predicted gain to step {es['future_step']}≈"
           f"{'n/a' if predicted_gain != predicted_gain else f'{predicted_gain:.4f}'} "
           f"(threshold {es['predicted_gain']})")
    return should_stop, status, msg


def train(cfg: RunConfig, resume: Optional[str], is_smoke: bool,
          metrics_file: Optional[str] = None, metrics_format: str = "jsonl",
          early_stop_policy: str = "off", es: Optional[dict] = None) -> dict:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as acc_set_seed

    set_seed(cfg.train.seed)
    acc_set_seed(cfg.train.seed)

    # Resolve the full platform-aware runtime against the machine actually present. Feature
    # flags that are unavailable are safely disabled (and reported below).
    resolved = rt.resolve_runtime(cfg.runtime, cfg.train.precision)
    rt.apply_backend_flags(resolved)
    precision = resolved.precision

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=precision_to_accelerate(precision),
        cpu=(resolved.device == "cpu"),
    )

    model = BertForMaskedLM(cfg.model)
    model = rt.maybe_compile(model, resolved)  # no-op unless torch_compile explicitly enabled
    param_report = parameter_count_report(model)
    param_count = count_parameters(model)["unique"]

    # Optional experiment tracker (default backend 'none' -> NullTracker no-op). Built before
    # the training loop so finish() can always be called in the finally block below.
    tracker = tk.build_tracker(cfg.tracking)
    if accelerator.is_main_process:
        startup_report(cfg, precision, str(accelerator.device), cfg.train.seed, param_report)
        print("-" * 74)
        print("resolved runtime (settings that materially affect training):")
        for line in rt.runtime_report_lines(resolved):
            print("  " + line)
        print("-" * 74)
        _write_run_artifacts(cfg, resolved, str(accelerator.device))
        if cfg.tracking.backend != "none":
            run_name = _run_name_for(cfg, resolved, param_count)
            tracker.init_run(
                config=_tracking_context(cfg, resolved, param_count, str(accelerator.device)),
                run_name=run_name, project=cfg.tracking.project, entity=cfg.tracking.entity,
                group=cfg.tracking.group, job_type=cfg.tracking.job_type,
                tags=list(cfg.tracking.tags), notes=cfg.tracking.notes,
                mode=cfg.tracking.mode, dir=cfg.output.dir,
            )
            print(f"[tracking] backend={cfg.tracking.backend} mode={cfg.tracking.mode} "
                  f"run={run_name}")

    train_loader, val_loader, specials = build_dataloaders(cfg.model, cfg.train, cfg.data,
                                                           runtime=resolved)
    optimizer = build_optimizer(model, cfg.train, extra_adamw_kwargs=rt.adamw_extra_kwargs(resolved))
    scheduler = build_scheduler(optimizer, cfg.train)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = 0
    if resume:
        resume_path = resolve_checkpoint_path(resume)
        try:
            payload = load_checkpoint(
                resume_path, model=accelerator.unwrap_model(model), optimizer=optimizer,
                scheduler=scheduler, map_location=accelerator.device, restore_rng=True,
                verify_checksum=True,
            )
        except CheckpointError as e:
            # e.g. a legacy checkpoint without a stored sha256; proceed without verification.
            print(f"[resume] checksum unavailable/failed ({e}); loading without verification")
            payload = load_checkpoint(
                resume_path, model=accelerator.unwrap_model(model), optimizer=optimizer,
                scheduler=scheduler, map_location=accelerator.device, restore_rng=True,
                verify_checksum=False,
            )
        global_step = int(payload.get("global_step", 0))
        if accelerator.is_main_process:
            print(f"[resume] restored global_step={global_step} from {resume_path}")

    os.makedirs(cfg.output.checkpoint_dir, exist_ok=True)
    ckpt_dir = cfg.output.checkpoint_dir
    ckpt_mgr = CheckpointManager(ckpt_dir)
    best_ckpt_val = float("inf")

    model.train()
    t0 = time.time()
    tokens_seen = 0
    last_metrics: dict = {}
    last_grad_norm: Optional[float] = None
    last_train_loss: float = float("nan")
    history: list[dict] = []
    metric_records: list[dict] = []  # analyzer-shaped rows (for metrics file + early stop)
    stop_reason: Optional[str] = None
    done = False

    # Instantaneous-rate tracking between log points.
    last_log_t, last_log_tokens, last_log_step = t0, 0, 0

    try:
        while not done:
            for batch in train_loader:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and cfg.train.max_grad_norm > 0:
                        gnorm = accelerator.clip_grad_norm_(model.parameters(),
                                                            cfg.train.max_grad_norm)
                        last_grad_norm = float(gnorm) if gnorm is not None else None
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                tokens_seen += int(batch["attention_mask"].sum())
                last_train_loss = float(loss.detach())

                if accelerator.sync_gradients:
                    global_step += 1

                    if global_step % cfg.train.log_every == 0 and accelerator.is_main_process:
                        lr = scheduler.get_last_lr()[0]
                        now = time.time()
                        tps = tokens_seen / max(1e-6, (now - t0))
                        dt = max(1e-6, now - last_log_t)
                        inst_tps = (tokens_seen - last_log_tokens) / dt
                        inst_sps = (global_step - last_log_step) / dt
                        last_log_t, last_log_tokens, last_log_step = now, tokens_seen, global_step
                        print(f"step {global_step:>5} | loss {last_train_loss:.4f} "
                              f"| lr {lr:.3e} | {tps:,.0f} tok/s")
                        train_log = {
                            "train/loss": last_train_loss,
                            "train/learning_rate": lr,
                            "train/tokens_seen": tokens_seen,
                            "train/tokens_per_second": inst_tps,
                            "train/steps_per_second": inst_sps,
                        }
                        if last_grad_norm is not None:
                            train_log["train/gradient_norm"] = last_grad_norm
                        if tracker.is_active:
                            tracker.log_metrics(train_log, step=global_step)

                    if global_step % cfg.train.eval_every == 0:
                        metrics = evaluate(model, val_loader, accelerator,
                                           cfg.train.eval_max_batches)
                        last_metrics = metrics
                        history.append({"step": global_step, **metrics})
                        if accelerator.is_main_process:
                            print(f"[eval] step {global_step} | val_loss "
                                  f"{metrics['val_loss']:.4f} | masked_acc "
                                  f"{metrics['val_masked_accuracy']:.4f} | ppl "
                                  f"{metrics['val_perplexity']:.2f}")
                            if tracker.is_active:
                                tracker.log_metrics({
                                    "eval/loss": metrics["val_loss"],
                                    "eval/perplexity": metrics["val_perplexity"],
                                    "eval/masked_accuracy": metrics["val_masked_accuracy"],
                                    "eval/masked_tokens": metrics["val_masked_tokens"],
                                }, step=global_step)

                        # Record an analyzer-shaped metrics row (and optionally persist it).
                        row = {
                            "step": global_step,
                            "tokens_seen": tokens_seen,
                            "train_loss": last_train_loss,
                            "val_loss": metrics["val_loss"],
                            "masked_accuracy": metrics["val_masked_accuracy"],
                            "learning_rate": scheduler.get_last_lr()[0],
                            "gradient_norm": last_grad_norm,
                        }
                        metric_records.append(row)
                        if metrics_file and accelerator.is_main_process:
                            _append_metrics_row(metrics_file, row, metrics_format)

                        # Early-stop policy (default 'off'): never auto-stops unless 'stop'.
                        if early_stop_policy != "off" and es is not None:
                            should_stop, status, msg = _early_stop_decision(metric_records, es)
                            if accelerator.is_main_process:
                                print(f"[early-stop:{early_stop_policy}] {msg}")
                            if early_stop_policy == "warn" and should_stop and \
                                    accelerator.is_main_process:
                                print("[early-stop:warn] RECOMMENDATION: run appears saturated; "
                                      "consider stopping (not auto-stopping; policy=warn).")
                            if early_stop_policy == "stop" and should_stop:
                                stop_reason = (f"early-stop policy triggered at step "
                                               f"{global_step}: {msg}")
                                if accelerator.is_main_process:
                                    print(f"[early-stop:stop] {stop_reason} "
                                          "-> saving final checkpoint and stopping.")
                                done = True
                                break

                    if global_step % cfg.train.save_every == 0 and accelerator.is_main_process:
                        best_ckpt_val = _do_checkpoint(
                            ckpt_mgr, accelerator, model, optimizer, scheduler, global_step,
                            cfg, precision, last_metrics, best_ckpt_val)

                    if global_step >= cfg.train.max_steps:
                        done = True
                        break

        # Final eval + checkpoint
        final_metrics = evaluate(model, val_loader, accelerator, cfg.train.eval_max_batches)
        last_metrics = final_metrics
        elapsed = time.time() - t0
        throughput = tokens_seen / max(1e-6, elapsed)
        mem = _peak_memory_mb()

        final_ckpt = None
        if accelerator.is_main_process:
            best_ckpt_val = _do_checkpoint(
                ckpt_mgr, accelerator, model, optimizer, scheduler, global_step, cfg, precision,
                final_metrics, best_ckpt_val)
            final_ckpt = ckpt_mgr.latest_path()
            print("-" * 74)
            print(f"[final] step {global_step} | val_loss {final_metrics['val_loss']:.4f} "
                  f"| masked_acc {final_metrics['val_masked_accuracy']:.4f} "
                  f"| ppl {final_metrics['val_perplexity']:.2f}")
            print(f"[final] elapsed {elapsed:.1f}s | {throughput:,.0f} tok/s "
                  f"| peak_mem {mem:.1f} MB | tokens_seen {tokens_seen:,}")
            print(f"[final] latest checkpoint: {final_ckpt}")
            print(f"[final] best checkpoint  : {ckpt_mgr.best_path()}")

            # System metrics + run analysis + summary + optional artifacts to the tracker.
            if tracker.is_active:
                _log_analysis_and_summary(
                    tracker, cfg, metric_records, final_metrics, ckpt_mgr, best_ckpt_val,
                    global_step, elapsed, tokens_seen, throughput, mem, stop_reason,
                    final_ckpt, metrics_file)

        result = {
            "global_step": global_step,
            "precision": precision,
            "device": str(accelerator.device),
            "elapsed_s": elapsed,
            "throughput_tok_s": throughput,
            "peak_mem_mb": mem,
            "tokens_seen": tokens_seen,
            "param_report": param_report,
            "history": history,
            "stop_reason": stop_reason,
            "early_stopped": stop_reason is not None,
            **{f"final_{k}": v for k, v in final_metrics.items()},
        }
    finally:
        # Always finish the tracker (even on exception). Never auto-sync offline runs.
        if accelerator.is_main_process and tracker.is_active:
            sc = tracker.sync_command
            if sc:
                print(f"[tracking] offline run dir : {tracker.run_dir}")
                print(f"[tracking] to upload later : {sc}")
                print("[tracking] (offline runs are NOT synced automatically)")
        tracker.finish()

    return result


def _log_analysis_and_summary(tracker, cfg, metric_records, final_metrics, ckpt_mgr,
                              best_ckpt_val, global_step, elapsed, tokens_seen, throughput,
                              mem, stop_reason, final_ckpt, metrics_file) -> Optional[str]:
    """Log system + analysis/* metrics, the run summary, and optional artifacts.

    Analysis reuses the project's own conservative curve analyzer over the in-memory metric
    rows (same source as the JSONL file) — it does not replace it.
    """
    # System metrics.
    sys_metrics = {"system/peak_ram_mb": mem}
    if torch.cuda.is_available():
        sys_metrics["system/peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    tracker.log_metrics(sys_metrics, step=global_step)

    # Conservative curve analysis (best-effort; never fails the run).
    analysis_status = None
    predicted_asymptote = None
    recommended_stop = None
    recent_improvement = None
    try:
        from coordinator_bert.curve_analysis import AnalysisConfig, analyze
        if len(metric_records) >= 4:
            res = analyze(metric_records, AnalysisConfig(run_id=cfg.tracking.project))
            analysis_status = res.status
            predicted_asymptote = res.predicted_asymptote.get("point")
            recommended_stop = res.recommended_stop_step
            recent_improvement = res.recent_improvement_per_100
            tracker.log_metrics({
                "analysis/predicted_asymptotic_loss": predicted_asymptote
                if predicted_asymptote is not None else float("nan"),
                "analysis/recent_improvement": recent_improvement
                if recent_improvement is not None else float("nan"),
                **({"analysis/recommended_stop_step": recommended_stop}
                   if recommended_stop is not None else {}),
            }, step=global_step)
    except Exception as e:  # noqa: BLE001
        print(f"[tracking] analysis logging skipped: {e}")

    best_step = None
    ptr = ckpt_mgr.read_pointer() or {}
    if ptr.get("best"):
        try:
            best_step = int(ptr["best"].split("_")[1])
        except (IndexError, ValueError):
            best_step = None

    summary = {
        "final_global_step": global_step,
        "final_train_loss": metric_records[-1]["train_loss"] if metric_records else None,
        "final_val_loss": final_metrics.get("val_loss"),
        "final_val_perplexity": final_metrics.get("val_perplexity"),
        "best_val_loss": best_ckpt_val if best_ckpt_val != float("inf") else
        final_metrics.get("val_loss"),
        "best_step": best_step,
        "final_masked_accuracy": final_metrics.get("val_masked_accuracy"),
        "elapsed_seconds": elapsed,
        "total_tokens": tokens_seen,
        "mean_tokens_per_second": throughput,
        "stop_reason": stop_reason,
        "checkpoint_path": final_ckpt,
        "analysis_status": analysis_status,
    }
    tracker.log_summary(summary)

    # Optional artifacts (never upload datasets/secrets; local checkpoints stay authoritative).
    if cfg.tracking.log_analysis_artifacts:
        out_dir = cfg.output.dir
        for fname, atype, aname in (
            ("resolved_config.yaml", "config", "resolved_config"),
            ("environment.json", "config", "environment"),
        ):
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                tracker.log_artifact(p, aname, atype)
        if metrics_file and os.path.exists(metrics_file):
            tracker.log_artifact(metrics_file, "metrics", "metrics")
    # Selected final/best checkpoint only when explicitly enabled.
    if cfg.tracking.log_checkpoints:
        best_ckpt = ckpt_mgr.best_path() or final_ckpt
        if best_ckpt and os.path.isdir(best_ckpt):
            tracker.log_artifact(best_ckpt, "model", "model")

    return analysis_status


def _do_checkpoint(ckpt_mgr, accelerator, model, optimizer, scheduler, global_step, cfg,
                   precision, last_metrics, best_ckpt_val) -> float:
    """Save an immutable step_XXXXXX checkpoint and update latest/best pointers.

    Returns the (possibly updated) best checkpoint val loss. "best" is a pointer only — no
    300+ MB duplication. A single write per checkpoint (no 'last' mirror copy).
    """
    path = ckpt_mgr.save(
        global_step,
        model=accelerator.unwrap_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg.model,
        precision=precision,
        device=str(accelerator.device),
        metadata={"seed": cfg.train.seed},
    )
    val = last_metrics.get("val_loss") if isinstance(last_metrics, dict) else None
    if val is not None and val < best_ckpt_val:
        best_ckpt_val = float(val)
        ckpt_mgr.mark_best(global_step, metric=best_ckpt_val)
    print(f"[checkpoint] step {global_step} -> {os.path.basename(path)} "
          f"(latest{'; best' if val is not None and val <= best_ckpt_val else ''})")
    return best_ckpt_val


def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    try:
        import resource
        # ru_maxrss is KB on Linux, bytes on macOS.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 if sys.platform != "darwin" else rss / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLM pretraining for coordinator_bert.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--smoke", action="store_true", help="Run a short self-contained smoke run.")
    p.add_argument("--resume", default=None, help="Checkpoint dir to resume from.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override train.max_steps (useful for staged smoke/resume runs).")
    p.add_argument("--eval-every", type=int, default=None, help="Override train.eval_every.")
    p.add_argument("--seed", type=int, default=None, help="Override train.seed.")
    # Metrics logging (optional; independent of early stop).
    p.add_argument("--metrics-file", default=None,
                   help="Write per-eval metrics to this JSONL/CSV file (format from extension).")
    # Early-stop policy. DEFAULT 'off' — never auto-stops unless explicitly set to 'stop'.
    p.add_argument("--early-stop-policy", choices=["off", "warn", "stop"], default="off",
                   help="off (default): no analysis. warn: print recommendations only. "
                        "stop: may stop after guards are met (always saves a final checkpoint).")
    p.add_argument("--es-patience", type=int, default=5)
    p.add_argument("--es-min-delta", type=float, default=1e-3)
    p.add_argument("--es-min-evals", type=int, default=8)
    p.add_argument("--es-min-gain-per-100", type=float, default=1e-3)
    p.add_argument("--es-predicted-gain", type=float, default=2e-2,
                   help="Stop only if predicted absolute loss gain to the horizon is below this.")
    p.add_argument("--es-future-step", type=int, default=2000,
                   help="Forecast horizon (steps ahead) used by the early-stop analysis.")
    # Optional experiment tracking (default backend from config, usually 'none').
    p.add_argument("--wandb", action="store_true",
                   help="Enable the W&B tracking backend (overrides config tracking.backend).")
    p.add_argument("--wandb-mode", choices=["offline", "online", "disabled"], default=None,
                   help="Override tracking.mode (default offline).")
    p.add_argument("--wandb-project", default=None, help="Override tracking.project.")
    p.add_argument("--run-name", default=None, help="Override tracking.run_name.")
    p.add_argument("--wandb-log-checkpoints", action="store_true",
                   help="Also log the best/final checkpoint as a W&B model artifact.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    if args.max_steps is not None:
        cfg = replace(cfg, train=replace(cfg.train, max_steps=args.max_steps))
    if args.eval_every is not None:
        cfg = replace(cfg, train=replace(cfg.train, eval_every=args.eval_every))
    if args.seed is not None:
        cfg = replace(cfg, train=replace(cfg.train, seed=args.seed))

    # Optional tracking overrides (CLI wins over config). Default stays 'none'.
    tk_over = {}
    if args.wandb:
        tk_over["backend"] = "wandb"
    if args.wandb_mode is not None:
        tk_over["mode"] = args.wandb_mode
    if args.wandb_project is not None:
        tk_over["project"] = args.wandb_project
    if args.run_name is not None:
        tk_over["run_name"] = args.run_name
    if args.wandb_log_checkpoints:
        tk_over["log_checkpoints"] = True
    if tk_over:
        cfg = replace(cfg, tracking=replace(cfg.tracking, **tk_over))

    # Metrics file: use the given path, or default to one under the output dir when a policy
    # other than 'off' is active (so analysis has data to persist).
    metrics_file = args.metrics_file
    if metrics_file is None and (args.early_stop_policy != "off"
                                 or cfg.tracking.backend != "none"):
        metrics_file = os.path.join(cfg.output.dir, "metrics.jsonl")
    metrics_format = "csv" if (metrics_file or "").lower().endswith(".csv") else "jsonl"

    es = {
        "patience": args.es_patience, "min_delta": args.es_min_delta,
        "min_evals": args.es_min_evals, "min_gain_per_100": args.es_min_gain_per_100,
        "predicted_gain": args.es_predicted_gain, "future_step": args.es_future_step,
        "run_id": os.path.basename(cfg.output.dir.rstrip("/")) or "run",
    }

    result = train(cfg, resume=args.resume, is_smoke=args.smoke,
                   metrics_file=metrics_file, metrics_format=metrics_format,
                   early_stop_policy=args.early_stop_policy, es=es)
    print("[done]", {k: v for k, v in result.items() if k != "history"})


if __name__ == "__main__":
    main()
