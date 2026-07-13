#!/usr/bin/env python3
"""Tokenize and pack a real-text corpus into fixed-length token-ID shards (offline).

Packs ``[CLS] content [SEP] PAD...`` rows into ``data/tokenized/<run>/{train,validation}/*.npy``
with a manifest. Long documents are split across multiple sequences; a sequence never crosses a
document boundary. No MLM masks/labels are stored — masking stays dynamic at train time.

Example:
  python scripts/tokenize_and_pack_corpus.py \
    --tokenizer artifacts/tokenizers/bert-cord-byte-bpe-32k \
    --train-input data/tokenizer_corpus_en data/tokenizer_corpus_th \
    --validation-input data/tokenizer_corpus_en/validation.txt \
    --output-dir data/tokenized/bert_cord_en_th_128_v1 \
    --sequence-length 128 --sequences-per-shard 100000 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.packed_corpus import pack_corpus  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Tokenize and pack a corpus into token-ID shards.")
    p.add_argument("--tokenizer", required=True, help="Tokenizer dir or tokenizer.json.")
    p.add_argument("--train-input", nargs="+", required=True, help="Files/dirs (.txt).")
    p.add_argument("--validation-input", nargs="*", default=[], help="Files/dirs (.txt).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--sequences-per-shard", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--expected-vocab-size", type=int, default=32000)
    p.add_argument("--progress-every", type=int, default=10_000)
    args = p.parse_args()

    try:
        manifest = pack_corpus(
            args.tokenizer, args.train_input, args.validation_input, args.output_dir,
            sequence_length=args.sequence_length, sequences_per_shard=args.sequences_per_shard,
            seed=args.seed, overwrite=args.overwrite,
            expected_vocab_size=args.expected_vocab_size, progress_every=args.progress_every,
            project_root=_ROOT,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[pack] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    c = manifest["counters"]
    print("=" * 68)
    print(f"[pack] output dir       : {args.output_dir}")
    print(f"[pack] dtype / seq_len  : {manifest['dtype']} / {manifest['sequence_length']}")
    print(f"[pack] documents read   : {c['documents_read']:,} "
          f"(skipped {c['documents_skipped']:,})")
    print(f"[pack] source tokens    : {c['source_content_tokens']:,}")
    print(f"[pack] packed sequences : {c['packed_sequences']:,} "
          f"(train {manifest['train_sequences']:,}, val {manifest['validation_sequences']:,})")
    print(f"[pack] padding tokens   : {c['padding_tokens']:,} "
          f"| packing efficiency {c['packing_efficiency']:.2%}")
    print(f"[pack] unknown tokens   : {c['unknown_tokens']:,} "
          f"({c['unknown_token_rate']:.4%})")
    langs = {s["path"]: s["language"] for s in manifest["source_files"]}
    print(f"[pack] source languages : {langs}")
    print(f"[pack] tokenizer sha256 : {str(manifest['tokenizer_sha256'])[:16]}…")
    print(f"[pack] manifest         : {os.path.join(args.output_dir, 'manifest.json')}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
