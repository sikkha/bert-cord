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

from coordinator_bert.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from coordinator_bert.configuration import RunConfig, load_config  # noqa: E402
from coordinator_bert.data import build_dataloaders  # noqa: E402
from coordinator_bert.model import BertForMaskedLM, parameter_count_report  # noqa: E402


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


# --------------------------------------------------------------------------------------- #
# Optimizer / scheduler
# --------------------------------------------------------------------------------------- #
def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """AdamW with no weight decay on bias / LayerNorm parameters."""
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
    )


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
        max_steps=40,
        warmup_steps=5,
        eval_every=20,
        save_every=20,
        eval_max_batches=5,
        log_every=5,
        per_device_batch_size=8,
        gradient_accumulation_steps=2,
        max_seq_length=64,
    )
    syn = replace(cfg.data.synthetic, num_train_examples=512, num_val_examples=128,
                  max_len=64)
    data = replace(cfg.data, synthetic=syn, dataset_name=None)
    output = replace(
        cfg.output,
        dir="experiments/smoke",
        checkpoint_dir="experiments/smoke/checkpoints",
    )
    return replace(cfg, train=train, data=data, output=output)


def train(cfg: RunConfig, resume: Optional[str], is_smoke: bool) -> dict:
    from accelerate import Accelerator
    from accelerate.utils import set_seed as acc_set_seed

    set_seed(cfg.train.seed)
    acc_set_seed(cfg.train.seed)

    precision = resolve_precision(cfg.train.precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=precision_to_accelerate(precision),
    )

    model = BertForMaskedLM(cfg.model)
    param_report = parameter_count_report(model)
    if accelerator.is_main_process:
        startup_report(cfg, precision, str(accelerator.device), cfg.train.seed, param_report)

    train_loader, val_loader, specials = build_dataloaders(cfg.model, cfg.train, cfg.data)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = 0
    if resume:
        payload = load_checkpoint(
            resume,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=accelerator.device,
            restore_rng=True,
        )
        global_step = int(payload.get("global_step", 0))
        if accelerator.is_main_process:
            print(f"[resume] restored global_step={global_step} from {resume}")

    os.makedirs(cfg.output.checkpoint_dir, exist_ok=True)
    ckpt_dir = cfg.output.checkpoint_dir

    model.train()
    t0 = time.time()
    tokens_seen = 0
    last_metrics: dict = {}
    history: list[dict] = []
    done = False

    while not done:
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients and cfg.train.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            tokens_seen += int(batch["attention_mask"].sum())

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % cfg.train.log_every == 0 and accelerator.is_main_process:
                    lr = scheduler.get_last_lr()[0]
                    tps = tokens_seen / max(1e-6, (time.time() - t0))
                    print(f"step {global_step:>5} | loss {float(loss.detach()):.4f} "
                          f"| lr {lr:.3e} | {tps:,.0f} tok/s")

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

                if global_step % cfg.train.save_every == 0 and accelerator.is_main_process:
                    _save(accelerator, model, optimizer, scheduler, global_step, cfg,
                          ckpt_dir, precision)

                if global_step >= cfg.train.max_steps:
                    done = True
                    break

    # Final eval + checkpoint
    final_metrics = evaluate(model, val_loader, accelerator, cfg.train.eval_max_batches)
    last_metrics = final_metrics
    elapsed = time.time() - t0
    throughput = tokens_seen / max(1e-6, elapsed)
    mem = _peak_memory_mb()

    if accelerator.is_main_process:
        _save(accelerator, model, optimizer, scheduler, global_step, cfg, ckpt_dir,
              precision, tag="last")
        print("-" * 74)
        print(f"[final] step {global_step} | val_loss {final_metrics['val_loss']:.4f} "
              f"| masked_acc {final_metrics['val_masked_accuracy']:.4f} "
              f"| ppl {final_metrics['val_perplexity']:.2f}")
        print(f"[final] elapsed {elapsed:.1f}s | {throughput:,.0f} tok/s "
              f"| peak_mem {mem:.1f} MB | tokens_seen {tokens_seen:,}")
        print(f"[final] checkpoint: {os.path.join(ckpt_dir, 'last')}")

    return {
        "global_step": global_step,
        "precision": precision,
        "device": str(accelerator.device),
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
        "peak_mem_mb": mem,
        "tokens_seen": tokens_seen,
        "param_report": param_report,
        "history": history,
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }


def _save(accelerator, model, optimizer, scheduler, global_step, cfg, ckpt_dir,
          precision, tag: Optional[str] = None) -> None:
    tag = tag or f"step_{global_step}"
    for name in (tag, "last"):
        path = os.path.join(ckpt_dir, name)
        save_checkpoint(
            path,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            config=cfg.model,
            metadata={"precision": precision, "seed": cfg.train.seed},
        )
    print(f"[checkpoint] saved step {global_step} -> {os.path.join(ckpt_dir, tag)} (+last)")


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    result = train(cfg, resume=args.resume, is_smoke=args.smoke)
    print("[done]", {k: v for k, v in result.items() if k != "history"})


if __name__ == "__main__":
    main()
