#!/usr/bin/env python3
"""Evaluate a checkpoint on UNSEEN synthetic motif sequences.

For each (motif period, sequence length) combination, a fresh set of synthetic sequences is
generated with an eval-only seed (disjoint from any training seed), masked, and scored. Reports
MLM loss, top-1 and top-5 masked-token accuracy per combination and overall.

Milestone 0.5 scope: this measures *synthetic generalization* of the learned copy rule across
periods/lengths. It is not a claim about language understanding.

Example:
  python scripts/evaluate_synthetic.py --config configs/bert_25m.yaml \
      --checkpoint experiments/smoke/checkpoints/last --periods 2 3 4 --seq-lens 24 48
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.configuration import load_config  # noqa: E402
from coordinator_bert.data import SpecialTokens, build_synthetic_examples  # noqa: E402
from coordinator_bert.inference import (  # noqa: E402
    load_model_for_inference,
    masked_accuracy_topk,
)
from coordinator_bert.masking import MLMasker  # noqa: E402


def _parse_int_list(text) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    return [int(x) for x in str(text).replace(",", " ").split()]


def build_eval_batch(cfg, specials, num_examples, seq_len, period, seed, mlm_probability):
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
    g = torch.Generator().manual_seed(seed)
    out = masker(input_ids, special_mask=special_mask, generator=g)
    return {"input_ids": out.input_ids, "attention_mask": attention_mask,
            "labels": out.labels}


@torch.no_grad()
def evaluate_combo(model, batch, vocab_size) -> dict:
    logits = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"])["logits"]
    labels = batch["labels"]
    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    acc = masked_accuracy_topk(logits, labels, ks=(1, 5))
    return {
        "loss": float(loss),
        "perplexity": math.exp(min(20.0, float(loss))),
        "top1": acc[1],
        "top5": acc[5],
        "masked_tokens": int((labels != -100).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate on unseen synthetic motif sequences.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint dir. If omitted, uses random-init weights (baseline).")
    p.add_argument("--periods", nargs="+", default=["2", "3", "4"])
    p.add_argument("--seq-lens", nargs="+", default=["24", "48"])
    p.add_argument("--num-examples", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260711,
                   help="Eval-only seed base (disjoint from training seeds).")
    p.add_argument("--mlm-probability", type=float, default=0.15)
    args = p.parse_args()

    cfg = load_config(args.config)
    specials = SpecialTokens()
    periods = _parse_int_list(args.periods)
    seq_lens = _parse_int_list(args.seq_lens)

    model = load_model_for_inference(cfg.model, args.checkpoint)
    ckpt_desc = args.checkpoint or "(random-init)"
    print(f"[evaluate_synthetic] checkpoint={ckpt_desc}")
    print(f"[evaluate_synthetic] periods={periods} seq_lens={seq_lens} "
          f"num_examples={args.num_examples}")
    print(f"{'period':>6} {'seq_len':>7} {'loss':>8} {'ppl':>10} {'top1':>7} "
          f"{'top5':>7} {'masked':>7}")

    rows = []
    combo_seed = args.seed
    for period in periods:
        for seq_len in seq_lens:
            combo_seed += 1  # distinct unseen data per combo
            batch = build_eval_batch(cfg, specials, args.num_examples, seq_len, period,
                                     combo_seed, args.mlm_probability)
            m = evaluate_combo(model, batch, cfg.model.vocab_size)
            rows.append(m)
            print(f"{period:>6} {seq_len:>7} {m['loss']:>8.4f} {m['perplexity']:>10.2f} "
                  f"{m['top1']:>7.4f} {m['top5']:>7.4f} {m['masked_tokens']:>7}")

    n = len(rows)
    avg = {
        "loss": sum(r["loss"] for r in rows) / n,
        "top1": sum(r["top1"] for r in rows) / n,
        "top5": sum(r["top5"] for r in rows) / n,
    }
    print("-" * 56)
    print(f"[evaluate_synthetic] overall: loss {avg['loss']:.4f} | "
          f"top1 {avg['top1']:.4f} | top5 {avg['top5']:.4f} over {n} combos")


if __name__ == "__main__":
    main()
