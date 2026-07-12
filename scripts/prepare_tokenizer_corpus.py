#!/usr/bin/env python3
"""Prepare a tokenizer-training corpus from local files and/or a Hugging Face dataset.

Reads .txt / .md / .jsonl inputs (dirs are recursed), normalizes Unicode, drops empties,
exact-deduplicates, deterministically shuffles, computes language statistics, and writes a
sharded corpus + manifest + report.

Example:
  python scripts/prepare_tokenizer_corpus.py \
    --input data/raw \
    --output-dir data/tokenizer_corpus \
    --normalization NFC --val-fraction 0.02 --shard-size 100000

  # optional Hugging Face source (streamed, capped):
  python scripts/prepare_tokenizer_corpus.py --hf-dataset wikitext \
    --hf-config wikitext-2-raw-v1 --hf-split train --hf-max-docs 20000 \
    --output-dir data/tokenizer_corpus
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.corpus import CorpusConfig, prepare_corpus  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare a tokenizer-training corpus.")
    p.add_argument("--input", nargs="*", default=[], help="Files/dirs (.txt/.md/.jsonl).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--normalization", default="NFC",
                   choices=["NFC", "NFKC", "NFD", "NFKD", "none"])
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--min-chars", type=int, default=1)
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--text-field", default="text", help="Field for jsonl / HF datasets.")
    # Optional HF dataset source.
    p.add_argument("--hf-dataset", default=None)
    p.add_argument("--hf-config", default=None)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-max-docs", type=int, default=None)
    args = p.parse_args()

    if not args.input and not args.hf_dataset:
        print("[corpus] nothing to do: provide --input and/or --hf-dataset", file=sys.stderr)
        return 2

    cfg = CorpusConfig(
        normalization=args.normalization, dedup=not args.no_dedup, min_chars=args.min_chars,
        shuffle_seed=args.shuffle_seed, val_fraction=args.val_fraction,
        shard_size=args.shard_size, text_field=args.text_field,
    )
    hf = None
    if args.hf_dataset:
        hf = {"name": args.hf_dataset, "config": args.hf_config, "split": args.hf_split,
              "max_docs": args.hf_max_docs}

    try:
        manifest = prepare_corpus(args.input, args.output_dir, cfg, hf_dataset=hf)
    except Exception as e:  # noqa: BLE001
        print(f"[corpus] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    st = manifest["statistics"]
    print("=" * 64)
    print(f"[corpus] output dir     : {args.output_dir}")
    print(f"[corpus] documents kept : {st['documents_kept']:,} "
          f"(read {st['documents_read']:,}, dup -{st['duplicates_removed']:,}, "
          f"empty -{st['empty_removed']:,})")
    print(f"[corpus] train docs     : {manifest['train_documents']:,} in "
          f"{len(manifest['train_shards'])} shard(s)")
    print(f"[corpus] validation docs: {manifest['validation_documents']:,}")
    print(f"[corpus] languages      : "
          f"{dict(sorted(st['per_language_docs'].items(), key=lambda x: -x[1]))}")
    print(f"[corpus] manifest       : {os.path.join(args.output_dir, 'corpus_manifest.json')}")
    print(f"[corpus] report         : {os.path.join(args.output_dir, 'corpus_report.md')}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
