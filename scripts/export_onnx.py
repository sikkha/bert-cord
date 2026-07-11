#!/usr/bin/env python3
"""Export a trained BertForMaskedLM checkpoint to a portable ONNX inference artifact.

Exports the MLM inference graph (input_ids, attention_mask, token_type_ids -> logits) with
dynamic batch and sequence axes. Optimizer/scheduler/RNG/loss/labels are NOT exported — the
PyTorch checkpoint remains the training source of truth.

Example:
  python scripts/export_onnx.py \
    --config configs/bert_25m_mac.yaml \
    --checkpoint experiments/smoke/checkpoints \
    --output exports/bert_cord_27m_mlm.onnx \
    --sequence-length 128
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.checkpointing import resolve_checkpoint_path  # noqa: E402
from coordinator_bert.configuration import load_config  # noqa: E402
from coordinator_bert.onnx_export import (  # noqa: E402
    DEFAULT_OPSET,
    OnnxDependencyError,
    export_checkpoint_to_onnx,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Export BertForMaskedLM to ONNX (MLM inference).")
    p.add_argument("--config", required=True, help="YAML config (model dims must match ckpt).")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint root (latest.json), step dir, or state.pt. "
                        "If omitted, random-init weights are exported (debug only).")
    p.add_argument("--output", required=True, help="Output .onnx path.")
    p.add_argument("--sequence-length", type=int, default=128,
                   help="Example sequence length for tracing (axis stays dynamic).")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Example batch size for tracing (axis stays dynamic).")
    p.add_argument("--static", action="store_true",
                   help="Disable dynamic axes (fixed batch/seq). Default: dynamic.")
    args = p.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"[export_onnx] failed to load config {args.config}: {e}", file=sys.stderr)
        return 2

    resolved = args.checkpoint
    if args.checkpoint is not None:
        resolved = resolve_checkpoint_path(args.checkpoint)
        if not os.path.exists(resolved):
            print(f"[export_onnx] checkpoint not found: {args.checkpoint} "
                  f"(resolved: {resolved})", file=sys.stderr)
            return 2

    print(f"[export_onnx] config           : {args.config}")
    print(f"[export_onnx] checkpoint (in)   : {args.checkpoint}")
    print(f"[export_onnx] checkpoint (used) : {resolved}")
    print(f"[export_onnx] output            : {args.output}")
    print(f"[export_onnx] opset             : {args.opset}")
    print(f"[export_onnx] seq_length (trace): {args.sequence_length} "
          f"(dynamic={not args.static})")

    try:
        meta = export_checkpoint_to_onnx(
            cfg.model, args.checkpoint, args.output,
            sequence_length=args.sequence_length, opset=args.opset,
            batch_size=args.batch_size, dynamic=not args.static,
        )
    except OnnxDependencyError as e:
        print(f"[export_onnx] {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"[export_onnx] export FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    mb = meta["size_bytes"] / (1024 ** 2)
    print("-" * 60)
    print(f"[export_onnx] SUCCESS")
    print(f"[export_onnx] input names  : {meta['input_names']}")
    print(f"[export_onnx] output names : {meta['output_names']}")
    print(f"[export_onnx] params       : {meta['param_count']:,} (~{meta['param_count']/1e6:.2f}M)")
    print(f"[export_onnx] graph size   : {meta['graph_size_bytes']:,} bytes")
    if meta["external_data_files"]:
        print(f"[export_onnx] external data : {meta['external_data_bytes']:,} bytes "
              f"in {[os.path.basename(p) for p in meta['external_data_files']]} "
              "(ship alongside the .onnx)")
    print(f"[export_onnx] TOTAL artifact: {meta['size_bytes']:,} bytes ({mb:.2f} MB)")
    print(f"[export_onnx] wrote        : {meta['output_path']}")
    print("[export_onnx] structural validation (onnx.checker): PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
