#!/usr/bin/env python3
"""Overfit-capacity check: memorize a fixed tiny synthetic batch.

Generates a small, fixed synthetic dataset and a single fixed masked batch, then trains on
that same batch until the model overfits it. Reports loss and masked-token accuracy and
**exits non-zero if final masked top-1 accuracy does not exceed a threshold**.

This validates that the gradient/optimizer path has the *capacity* to fit data — a standard
sanity check. It says nothing about language understanding or generalization.

Example:
  python scripts/overfit_tiny.py --config configs/bert_25m.yaml --steps 200 --threshold 0.95
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from coordinator_bert.configuration import load_config  # noqa: E402
from coordinator_bert.data import SpecialTokens, build_synthetic_examples  # noqa: E402
from coordinator_bert.inference import masked_accuracy_topk  # noqa: E402
from coordinator_bert.masking import MLMasker  # noqa: E402
from coordinator_bert.model import BertForMaskedLM, parameter_count_report  # noqa: E402

from pretrain_mlm import build_optimizer, set_seed  # noqa: E402


def build_fixed_batch(cfg, specials, num_examples, seq_len, period, seed, mlm_probability):
    """Build one fixed, padded, masked batch that the model will try to memorize."""
    examples = build_synthetic_examples(
        num_examples=num_examples, vocab_size=cfg.model.vocab_size, specials=specials,
        min_len=seq_len, max_len=seq_len, seed=seed, period=period,
    )
    max_len = max(len(e) for e in examples)
    input_ids = torch.full((len(examples), max_len), specials.pad, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, e in enumerate(examples):
        input_ids[i, : len(e)] = torch.tensor(e, dtype=torch.long)
        attention_mask[i, : len(e)] = 1

    masker = MLMasker(
        mask_token_id=specials.mask, vocab_size=cfg.model.vocab_size,
        special_token_ids=specials.all_ids, mlm_probability=mlm_probability,
    )
    special_mask = masker.special_tokens_mask(input_ids) | (attention_mask == 0)
    g = torch.Generator().manual_seed(seed + 999)
    out = masker(input_ids, special_mask=special_mask, generator=g)
    return {"input_ids": out.input_ids, "attention_mask": attention_mask,
            "labels": out.labels}


def main() -> int:
    p = argparse.ArgumentParser(description="Overfit a fixed tiny synthetic batch.")
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--threshold", type=float, default=0.9,
                   help="Fail unless final masked top-1 accuracy exceeds this.")
    p.add_argument("--num-examples", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--period", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=20, help="Linear LR warmup steps (stabilizes post-LN).")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    specials = SpecialTokens()

    model = BertForMaskedLM(cfg.model)
    model.train()
    print(parameter_count_report(model))

    batch = build_fixed_batch(cfg, specials, args.num_examples, args.seq_len, args.period,
                              args.seed, args.mlm_probability)
    n_masked = int((batch["labels"] != -100).sum())
    print(f"[overfit] fixed batch: {batch['input_ids'].shape[0]} seqs x "
          f"{batch['input_ids'].shape[1]} tok | {n_masked} masked positions | "
          f"lr={args.lr} steps={args.steps} threshold={args.threshold}")

    # Minimal TrainConfig-like object for build_optimizer (only needs a few attrs).
    class _Opt:
        weight_decay = 0.01
        learning_rate = args.lr
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-8

    optimizer = build_optimizer(model, _Opt)

    def lr_scale(step: int) -> float:
        # Linear warmup then constant — warmup is the key stabilizer for post-LN here.
        return min(1.0, step / max(1, args.warmup))

    t0 = time.time()
    final_loss = float("nan")
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale(step)
        out = model(**batch)
        loss = out["loss"]
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        final_loss = float(loss.detach())
        if step % args.log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                acc = masked_accuracy_topk(model(**batch)["logits"], batch["labels"],
                                           ks=(1, 5))
            model.train()
            print(f"step {step:>4} | loss {final_loss:.4f} | top1 {acc[1]:.3f} "
                  f"| top5 {acc[5]:.3f}")

    model.eval()
    with torch.no_grad():
        logits = model(**batch)["logits"]
        acc = masked_accuracy_topk(logits, batch["labels"], ks=(1, 5))
    elapsed = time.time() - t0
    print("-" * 60)
    print(f"[overfit] final loss {final_loss:.4f} | masked top1 {acc[1]:.4f} "
          f"| masked top5 {acc[5]:.4f} | {elapsed:.1f}s")

    passed = acc[1] > args.threshold
    print(f"[overfit] {'PASS' if passed else 'FAIL'}: "
          f"top1 {acc[1]:.4f} {'>' if passed else '<='} threshold {args.threshold}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
