#!/usr/bin/env python3
"""Train a WordPiece tokenizer with Hugging Face ``tokenizers`` (optional utility).

Milestone 0 smoke training uses synthetic token ids and does not require this tokenizer. It
is provided so a real corpus can be tokenized later (Milestone 1+). Special tokens are placed
at fixed ids matching ``coordinator_bert.data.SpecialTokens``:
    [PAD]=0, [CLS]=1, [SEP]=2, [MASK]=3, [UNK]=4.

Usage:
  python scripts/train_tokenizer.py --input corpus.txt --vocab-size 32000 \
      --output artifacts/tokenizer.json
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Text file(s) to train on.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output", default="artifacts/tokenizer.json")
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.trainers import WordPieceTrainer

    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.train(args.input, trainer)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tokenizer.save(args.output)
    print(f"[tokenizer] vocab_size={tokenizer.get_vocab_size()} saved -> {args.output}")
    for tok in special_tokens:
        print(f"  {tok:>7} -> id {tokenizer.token_to_id(tok)}")


if __name__ == "__main__":
    main()
