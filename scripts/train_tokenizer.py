#!/usr/bin/env python3
"""Train a BERT-Cord subword tokenizer (config-driven).

Two ways to run:

  * config-driven (recommended):
      python scripts/train_tokenizer.py \
        --config configs/tokenizer/bert_cord_wordpiece_32k.yaml \
        --input data/tokenizer_corpus/train-00000.txt \
        --output-dir artifacts/tokenizers

  * legacy flags (backward compatible; trains a WordPiece tokenizer):
      python scripts/train_tokenizer.py --input corpus.txt --vocab-size 32000 \
        --output artifacts/tokenizer.json

Supports byte-level BPE, Unigram, and WordPiece via ``configs/tokenizer/*.yaml``. Special tokens
are pinned to fixed ids ([PAD]=0, [CLS]=1, [SEP]=2, [MASK]=3, [UNK]=4) to match the MLM pipeline.
The output artifact directory contains tokenizer.json, tokenizer_config.json,
special_tokens_map.json, tokenizer_manifest.json, and README.md.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.tokenizer_train import (  # noqa: E402
    TokenizerTrainConfig,
    train_tokenizer,
)


def _resolve_inputs(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        if os.path.isdir(pat):
            files.extend(sorted(glob.glob(os.path.join(pat, "**", "*.txt"), recursive=True)))
        else:
            files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    return sorted(set(files))


def main() -> int:
    p = argparse.ArgumentParser(description="Train a BERT-Cord tokenizer (config-driven).")
    p.add_argument("--config", default=None, help="configs/tokenizer/*.yaml (recommended).")
    p.add_argument("--input", nargs="+", required=True,
                   help="Training text file(s)/dir(s)/globs (.txt).")
    p.add_argument("--output-dir", default="artifacts/tokenizers",
                   help="Artifact root; a <name>/ subdir is created (config mode).")
    p.add_argument("--corpus-manifest", default=None,
                   help="Optional corpus_manifest.json to record provenance.")
    # Legacy single-file mode.
    p.add_argument("--output", default=None,
                   help="Legacy: write a bare tokenizer.json here (WordPiece).")
    p.add_argument("--vocab-size", type=int, default=None, help="Legacy override.")
    p.add_argument("--min-frequency", type=int, default=None, help="Legacy override.")
    args = p.parse_args()

    inputs = _resolve_inputs(args.input)
    if not inputs:
        print(f"[tokenizer] no training files matched: {args.input}", file=sys.stderr)
        return 2

    # Legacy bare-output mode (kept for backward compatibility).
    if args.output and not args.config:
        return _legacy_train(inputs, args)

    if args.config:
        cfg = TokenizerTrainConfig.from_yaml(args.config)
    else:
        cfg = TokenizerTrainConfig()  # defaults: wordpiece 32k
    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    if args.min_frequency is not None:
        cfg.min_frequency = args.min_frequency

    out_dir = os.path.join(args.output_dir, cfg.name)
    print(f"[tokenizer] algorithm   : {cfg.algorithm}")
    print(f"[tokenizer] vocab_size  : {cfg.vocab_size}")
    print(f"[tokenizer] inputs      : {len(inputs)} file(s)")
    print(f"[tokenizer] output dir  : {out_dir}")
    try:
        manifest = train_tokenizer(cfg, inputs, out_dir,
                                   corpus_manifest=args.corpus_manifest, project_root=_ROOT)
    except Exception as e:  # noqa: BLE001
        print(f"[tokenizer] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("-" * 60)
    print(f"[tokenizer] SUCCESS — actual vocab {manifest['actual_vocab_size']:,}")
    print(f"[tokenizer] special ids : {manifest['special_token_ids']}")
    print(f"[tokenizer] sha256      : {manifest['tokenizer_json_sha256'][:16]}…")
    print(f"[tokenizer] wrote       : {out_dir}/"
          "{tokenizer.json, tokenizer_config.json, special_tokens_map.json, "
          "tokenizer_manifest.json, README.md}")
    return 0


def _legacy_train(inputs: list[str], args) -> int:
    """Original bare-tokenizer.json WordPiece path (kept for backward compatibility)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.trainers import WordPieceTrainer

    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.normalizer = BertNormalizer(lowercase=True)
    tok.pre_tokenizer = BertPreTokenizer()
    trainer = WordPieceTrainer(vocab_size=args.vocab_size or 32000,
                               min_frequency=args.min_frequency or 2,
                               special_tokens=special_tokens)
    tok.train(inputs, trainer)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tok.save(args.output)
    print(f"[tokenizer] (legacy) vocab_size={tok.get_vocab_size()} saved -> {args.output}")
    for t in special_tokens:
        print(f"  {t:>7} -> id {tok.token_to_id(t)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
