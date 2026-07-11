#!/usr/bin/env python3
"""Evaluate a trained MLM checkpoint: validation loss + masked-token accuracy.

Usage:
  python scripts/evaluate.py --config configs/bert_25m.yaml \
      --checkpoint experiments/smoke/checkpoints/last
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.checkpointing import load_checkpoint  # noqa: E402
from coordinator_bert.configuration import load_config  # noqa: E402
from coordinator_bert.data import build_dataloaders  # noqa: E402
from coordinator_bert.model import BertForMaskedLM, parameter_count_report  # noqa: E402


@torch.no_grad()
def run_eval(model, dataloader, max_batches: int) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_masked, n = 0.0, 0, 0, 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        out = model(**batch)
        total_loss += float(out["loss"])
        mask = batch["labels"] != -100
        preds = out["logits"].argmax(dim=-1)
        total_correct += int((preds[mask] == batch["labels"][mask]).sum())
        total_masked += int(mask.sum())
        n += 1
    avg = total_loss / max(1, n)
    return {
        "val_loss": avg,
        "val_masked_accuracy": total_correct / max(1, total_masked),
        "val_perplexity": math.exp(min(20.0, avg)),
        "val_masked_tokens": total_masked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MLM checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = BertForMaskedLM(cfg.model)
    load_checkpoint(args.checkpoint, model=model, map_location="cpu", restore_rng=False)
    print(parameter_count_report(model))

    _, val_loader, _ = build_dataloaders(cfg.model, cfg.train, cfg.data)
    metrics = run_eval(model, val_loader, args.max_batches)
    print(f"[eval] checkpoint={args.checkpoint}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
