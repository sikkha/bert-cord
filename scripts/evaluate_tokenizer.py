#!/usr/bin/env python3
"""Evaluate a trained BERT-Cord tokenizer over a corpus (intrinsic metrics).

Example:
  python scripts/evaluate_tokenizer.py \
    --tokenizer artifacts/tokenizers/bert-cord-wordpiece-32k \
    --input data/tokenizer_corpus/validation.txt \
    --output-dir artifacts/tokenizers/bert-cord-wordpiece-32k
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.tokenizer_eval import evaluate_tokenizer, write_reports  # noqa: E402


def _resolve(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        if os.path.isdir(pat):
            files.extend(sorted(glob.glob(os.path.join(pat, "**", "*.txt"), recursive=True)))
        else:
            files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    return sorted(set(files))


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a BERT-Cord tokenizer.")
    p.add_argument("--tokenizer", required=True, help="Tokenizer dir or tokenizer.json.")
    p.add_argument("--input", nargs="+", required=True, help="Eval text file(s)/dir(s)/globs.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write evaluation.json/.md (default: tokenizer dir).")
    p.add_argument("--max-lines", type=int, default=None)
    args = p.parse_args()

    eval_files = _resolve(args.input)
    if not eval_files:
        print(f"[eval] no eval files matched: {args.input}", file=sys.stderr)
        return 2

    try:
        metrics = evaluate_tokenizer(args.tokenizer, eval_files, max_lines=args.max_lines)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (args.tokenizer if os.path.isdir(args.tokenizer)
                                  else os.path.dirname(args.tokenizer))
    jpath, mpath = write_reports(metrics, out_dir)

    print("=" * 64)
    print(f"[eval] tokenizer         : {metrics['tokenizer_dir']}")
    print(f"[eval] vocab / util      : {metrics['vocab_size']:,} / "
          f"{metrics['vocabulary_utilization']:.2%}")
    print(f"[eval] unknown-token rate: {metrics['unknown_token_rate']:.4%}")
    print(f"[eval] tokens/sentence   : {metrics['avg_tokens_per_sentence']:.3f}")
    print(f"[eval] tokens/word       : {metrics['avg_tokens_per_word']:.3f}")
    print(f"[eval] round-trip (norm) : {metrics['roundtrip_normalized_rate']:.2%} "
          f"(exact {metrics['roundtrip_exact_rate']:.2%})")
    print(f"[eval] reserved integrity: {'OK' if metrics['reserved_token_integrity'] else 'FAIL'}")
    print(f"[eval] wrote             : {jpath}, {mpath}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
