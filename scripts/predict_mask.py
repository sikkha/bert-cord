#!/usr/bin/env python3
"""Masked-position prediction: load a checkpoint and print top-k predictions.

Two input modes:
  * explicit token ids:   --input "1 7 8 9 2"   (space/comma separated)
  * synthetic sequence:   (omit --input) a learnable motif sequence is generated from
                          --period / --seq-len / --seed

Masking:
  * --mask-positions "3 5"  masks those column indices; if omitted, the middle position is
    masked. Special tokens ([CLS]/[SEP]/[PAD]) are never chosen as default mask targets.

Milestone 0.5 scope: this validates inference mechanics only — it does not demonstrate
language understanding.

Example:
  python scripts/predict_mask.py --config configs/bert_25m.yaml \
      --checkpoint experiments/smoke/checkpoints/last --seq-len 24 --period 3 --topk 5
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.configuration import load_config  # noqa: E402
from coordinator_bert.data import SpecialTokens, build_synthetic_examples  # noqa: E402
from coordinator_bert.inference import (  # noqa: E402
    apply_mask_at,
    load_model_for_inference,
    predict_masked_topk,
)


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split()]


def _default_mask_position(ids: list[int], specials: SpecialTokens) -> int:
    """Pick the middle non-special position to mask when none is supplied."""
    interior = [i for i, t in enumerate(ids) if t not in specials.all_ids]
    if not interior:
        raise ValueError("Sequence has no maskable (non-special) tokens.")
    return interior[len(interior) // 2]


def main() -> None:
    p = argparse.ArgumentParser(description="Top-k masked-token prediction.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint dir. If omitted, uses random-init weights.")
    p.add_argument("--input", default=None, help="Explicit token ids (space/comma separated).")
    p.add_argument("--mask-positions", default=None,
                   help="Column indices to mask (space/comma separated).")
    p.add_argument("--period", type=int, default=3, help="Synthetic motif period.")
    p.add_argument("--seq-len", type=int, default=24, help="Synthetic sequence length.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    cfg = load_config(args.config)
    specials = SpecialTokens()
    torch.manual_seed(args.seed)

    # Build the input sequence.
    if args.input is not None:
        ids = _parse_int_list(args.input)
        source = "explicit token ids"
    else:
        ex = build_synthetic_examples(
            num_examples=1, vocab_size=cfg.model.vocab_size, specials=specials,
            min_len=args.seq_len, max_len=args.seq_len, seed=args.seed, period=args.period,
        )
        ids = ex[0]
        source = f"synthetic motif (period={args.period}, seq_len={len(ids)})"

    if args.mask_positions is not None:
        positions = _parse_int_list(args.mask_positions)
    else:
        positions = [_default_mask_position(ids, specials)]

    seq = torch.tensor(ids, dtype=torch.long)
    masked_seq, originals = apply_mask_at(seq, positions, specials.mask)

    model = load_model_for_inference(cfg.model, args.checkpoint)
    ckpt_desc = args.checkpoint or "(random-init)"
    print(f"[predict_mask] checkpoint={ckpt_desc}")
    print(f"[predict_mask] input source: {source}")
    print(f"[predict_mask] sequence ({len(ids)} tok): {ids}")
    print(f"[predict_mask] masked positions: {positions}")

    pos, top_ids, top_probs = predict_masked_topk(
        model, masked_seq, specials.mask, k=args.topk
    )
    for row_idx, (col, orig) in enumerate(zip(positions, originals.tolist())):
        preds = top_ids[row_idx].tolist()
        probs = top_probs[row_idx].tolist()
        hit = "OK" if preds and preds[0] == orig else "  "
        print(f"\n  position {col} | true token id = {orig} [{hit} top-1]")
        for rank, (tid, prob) in enumerate(zip(preds, probs), start=1):
            marker = " <- true" if tid == orig else ""
            print(f"    #{rank}: id={tid:<6} p={prob:.4f}{marker}")


if __name__ == "__main__":
    main()
