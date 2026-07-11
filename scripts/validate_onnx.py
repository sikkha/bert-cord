#!/usr/bin/env python3
"""Validate an exported ONNX model against the original PyTorch model (MLM parity).

Runs identical deterministic inputs through PyTorch and ONNX Runtime and checks: ONNX
structural validation, shape match, numerical closeness within documented FP32/CPU tolerances,
masked-position top-k agreement, attention-mask behavior, multiple sequence lengths and batch
sizes (dynamic axes), and absence of NaN/Inf. Exits non-zero if any check fails.

Tolerances (FP32 CPU): rtol=1e-3, atol=2e-3. ONNX export + ONNX Runtime may fuse/reorder
float ops and use different BLAS kernels than eager PyTorch, so bitwise-equal logits are not
expected; sub-2e-3 absolute differences on logits leave softmax/top-k unchanged. The decisive
correctness check is **exact top-k agreement** at masked positions (asserted == 1.0).

Example:
  python scripts/validate_onnx.py --config configs/bert_25m_mac.yaml \
      --checkpoint experiments/smoke/checkpoints --onnx-model exports/bert_cord_27m_mlm.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.configuration import load_config  # noqa: E402
from coordinator_bert.inference import load_model_for_inference  # noqa: E402
from coordinator_bert.onnx_export import (  # noqa: E402
    OnnxDependencyError,
    check_onnx_model,
    compare_logits,
    create_ort_session,
    example_inputs,
    run_onnx_logits,
    topk_agreement,
    torch_reference_logits,
)

RTOL = 1e-3
ATOL = 2e-3


def _interior_positions(batch: int, seq: int, k_positions: int = 3):
    """A few (row, col) positions per row, skipping the [CLS]/[SEP] ends."""
    cols = [c for c in range(1, seq - 1)]
    step = max(1, len(cols) // k_positions)
    chosen = cols[::step][:k_positions] or [seq // 2]
    return [(r, c) for r in range(batch) for c in chosen]


def main() -> int:
    p = argparse.ArgumentParser(description="PyTorch<->ONNX parity validation (MLM).")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--onnx-model", required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 64])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 3])
    args = p.parse_args()

    if not os.path.exists(args.onnx_model):
        print(f"[validate_onnx] onnx model not found: {args.onnx_model}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    try:
        check_onnx_model(args.onnx_model)
        print("[validate_onnx] ONNX structural validation (onnx.checker): PASSED")
        model = load_model_for_inference(cfg.model, args.checkpoint, map_location="cpu")
        session = create_ort_session(args.onnx_model, providers=["CPUExecutionProvider"])
    except OnnxDependencyError as e:
        print(f"[validate_onnx] {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"[validate_onnx] setup FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"[validate_onnx] active providers   : {session.get_providers()}")
    print(f"[validate_onnx] tolerances         : rtol={RTOL} atol={ATOL}")
    print(f"[validate_onnx] top-k agreement (k): {args.topk}")

    vocab = cfg.model.vocab_size
    pad = cfg.model.pad_token_id
    failures = []
    cases = []
    seed = 0
    # Cover multiple seq lengths x batch sizes; include one padded case to exercise the mask.
    for si, seq in enumerate(args.seq_lengths):
        for bi, batch in enumerate(args.batch_sizes):
            seed += 1
            pad_last = 2 if (si + bi) % 2 == 1 and seq > 6 else 0
            cases.append((batch, seq, pad_last, seed))

    print(f"[validate_onnx] cases (batch x seq x pad): "
          f"{[(b, s, pl) for b, s, pl, _ in cases]}")
    print("-" * 68)

    for batch, seq, pad_last, seed in cases:
        ii, am, tt = example_inputs(batch, seq, vocab, pad_token_id=pad, seed=seed,
                                    pad_last=pad_last)
        ref = torch_reference_logits(model, ii, am, tt)
        got = run_onnx_logits(session, ii, am, tt)
        stats = compare_logits(ref, got)
        positions = _interior_positions(batch, seq)
        agree = topk_agreement(ref, got, positions, k=args.topk)

        exp_shape = (batch, seq, vocab)
        ok_shape = stats["shape_a"] == exp_shape and stats["shapes_match"]
        ok_close = stats["max_abs_diff"] <= ATOL
        ok_topk = agree == 1.0
        ok_finite = not (stats["any_nan"] or stats["any_inf"])
        ok = ok_shape and ok_close and ok_topk and ok_finite
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] batch={batch} seq={seq} pad_last={pad_last} | "
              f"shape={stats['shape_a']} exp={exp_shape} | "
              f"max|Δ|={stats['max_abs_diff']:.2e} mean|Δ|={stats['mean_abs_diff']:.2e} | "
              f"top{args.topk}_agree={agree:.2f} | nan/inf={stats['any_nan'] or stats['any_inf']}")
        if not ok:
            reasons = []
            if not ok_shape:
                reasons.append(f"shape {stats['shape_a']} != {exp_shape}")
            if not ok_close:
                reasons.append(f"max_abs_diff {stats['max_abs_diff']:.2e} > atol {ATOL}")
            if not ok_topk:
                reasons.append(f"top-k agreement {agree:.2f} < 1.0")
            if not ok_finite:
                reasons.append("NaN/Inf present")
            failures.append(f"batch={batch} seq={seq}: " + "; ".join(reasons))

    # Dynamic-axes confirmation: at least two distinct seq lengths and two batch sizes ran.
    seqs = {s for _, s, _, _ in cases}
    batches = {b for b, _, _, _ in cases}
    dyn_ok = len(seqs) >= 2 and len(batches) >= 2
    print("-" * 68)
    print(f"[validate_onnx] dynamic axes exercised: seq_lengths={sorted(seqs)} "
          f"batch_sizes={sorted(batches)} -> {'OK' if dyn_ok else 'INSUFFICIENT'}")
    if not dyn_ok:
        failures.append("dynamic axes not exercised across >=2 seq lengths and >=2 batches")

    if failures:
        print(f"[validate_onnx] RESULT: FAIL ({len(failures)} issue(s))")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(f"[validate_onnx] RESULT: PASS — {len(cases)} cases, all within tolerance, "
          f"top-{args.topk} agreement 1.00, dynamic axes OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
