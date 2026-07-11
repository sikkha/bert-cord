#!/usr/bin/env python3
"""Masked-token prediction with ONNX Runtime (portable inference; MLM only).

Mirrors scripts/predict_mask.py where practical, but runs the exported ONNX graph via ONNX
Runtime instead of PyTorch. CPU execution by default. No tokenizer dependency — the project
still uses synthetic token ids.

Example:
  python scripts/predict_mask_onnx.py --model exports/bert_cord_27m_mlm.onnx \
      --period 3 --seq-len 24 --topk 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.data import SpecialTokens, build_synthetic_examples  # noqa: E402
from coordinator_bert.onnx_export import (  # noqa: E402
    OnnxDependencyError,
    create_ort_session,
    ort_input_names,
)


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split()]


def _default_mask_position(ids, specials) -> int:
    interior = [i for i, t in enumerate(ids) if t not in specials.all_ids]
    if not interior:
        raise ValueError("Sequence has no maskable (non-special) tokens.")
    return interior[len(interior) // 2]


def main() -> int:
    p = argparse.ArgumentParser(description="Top-k masked-token prediction via ONNX Runtime.")
    p.add_argument("--model", required=True, help="Path to the exported .onnx model.")
    p.add_argument("--input", default=None, help="Explicit token ids (space/comma separated).")
    p.add_argument("--mask-positions", default=None,
                   help="Column indices to mask (space/comma separated).")
    p.add_argument("--period", type=int, default=3, help="Synthetic motif period.")
    p.add_argument("--seq-len", type=int, default=24, help="Synthetic sequence length.")
    p.add_argument("--vocab-size", type=int, default=32000,
                   help="Vocab for synthetic generation (must match the exported model).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--providers", default=None,
                   help="Comma-separated ORT providers (default CPUExecutionProvider).")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"[predict_onnx] model not found: {args.model}", file=sys.stderr)
        return 2

    specials = SpecialTokens()

    # Build the input sequence (explicit ids or synthetic motif).
    if args.input is not None:
        ids = _parse_int_list(args.input)
        source = "explicit token ids"
    else:
        ex = build_synthetic_examples(
            num_examples=1, vocab_size=args.vocab_size, specials=specials,
            min_len=args.seq_len, max_len=args.seq_len, seed=args.seed, period=args.period,
        )
        ids = ex[0]
        source = f"synthetic motif (period={args.period}, seq_len={len(ids)})"

    positions = (_parse_int_list(args.mask_positions) if args.mask_positions is not None
                 else [_default_mask_position(ids, specials)])
    if any(c < 0 or c >= len(ids) for c in positions):
        print(f"[predict_onnx] mask position out of range for seq_len {len(ids)}: {positions}",
              file=sys.stderr)
        return 2

    ids = list(ids)
    originals = [ids[c] for c in positions]
    for c in positions:
        ids[c] = specials.mask

    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    token_type_ids = np.zeros_like(input_ids)

    try:
        providers = args.providers.split(",") if args.providers else None
        session = create_ort_session(args.model, providers=providers)
    except OnnxDependencyError as e:
        print(f"[predict_onnx] {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"[predict_onnx] failed to load model: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # Report providers and validate the input contract.
    import onnxruntime as ort  # already importable if session created
    print(f"[predict_onnx] model            : {args.model}")
    print(f"[predict_onnx] active providers  : {session.get_providers()}")
    print(f"[predict_onnx] available providers: {ort.get_available_providers()}")
    names = ort_input_names(session)
    print(f"[predict_onnx] model input names : {names}")
    expected = {"input_ids", "attention_mask", "token_type_ids"}
    if not expected.issubset(set(names)):
        print(f"[predict_onnx] model inputs {names} do not include {sorted(expected)}",
              file=sys.stderr)
        return 1

    feed = {"input_ids": input_ids, "attention_mask": attention_mask,
            "token_type_ids": token_type_ids}
    feed = {k: v for k, v in feed.items() if k in set(names)}
    logits = session.run(["logits"], feed)[0]

    if logits.ndim != 3:
        print(f"[predict_onnx] unexpected output rank {logits.ndim} (want 3)", file=sys.stderr)
        return 1
    print(f"[predict_onnx] output logits shape: {logits.shape}")
    print(f"[predict_onnx] input source       : {source}")
    print(f"[predict_onnx] sequence ({len(ids)} tok): {ids}")
    print(f"[predict_onnx] masked positions   : {positions}")

    for col, orig in zip(positions, originals):
        row = logits[0, col].astype(np.float64)
        probs = _softmax(row)
        top = np.argsort(-row)[: args.topk]
        hit = "OK" if int(top[0]) == orig else "  "
        print(f"\n  position {col} | true token id = {orig} [{hit} top-1]")
        for rank, tid in enumerate(top.tolist(), start=1):
            marker = " <- true" if tid == orig else ""
            print(f"    #{rank}: id={tid:<6} p={probs[tid]:.4f}{marker}")
    return 0


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


if __name__ == "__main__":
    raise SystemExit(main())
