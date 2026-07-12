"""Tokenizer evaluation metrics (Tokenizer Milestone).

Computes intrinsic tokenizer metrics over a text corpus: unknown-token rate, average tokens per
sentence and per word, round-trip decode fidelity, vocabulary utilization, and reserved-token
integrity. Dependency-light (``tokenizers`` + stdlib); no torch.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import unicodedata
from typing import Iterable, Optional

EXPECTED_SPECIAL_IDS = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4}
_WS = re.compile(r"\s+")


def _norm_ws(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _iter_lines(files: Iterable[str]):
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    yield line


def evaluate_tokenizer(tokenizer_dir: str, eval_files: list[str],
                       max_lines: Optional[int] = None) -> dict:
    """Evaluate the tokenizer at ``tokenizer_dir`` over ``eval_files``. Returns a metrics dict."""
    from tokenizers import Tokenizer

    tok_path = os.path.join(tokenizer_dir, "tokenizer.json") \
        if os.path.isdir(tokenizer_dir) else tokenizer_dir
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer.json not found at {tok_path}")
    tok = Tokenizer.from_file(tok_path)
    vocab_size = tok.get_vocab_size()
    unk_id = tok.token_to_id("[UNK]")

    total_tokens = 0        # content (non-special) subword tokens
    total_words = 0
    sentences = 0
    unk = 0
    used_ids: set = set()
    rt_exact = 0
    rt_norm = 0
    n_lines = 0

    for line in _iter_lines(eval_files):
        if max_lines is not None and n_lines >= max_lines:
            break
        n_lines += 1
        sentences += 1
        enc = tok.encode(line)
        mask = enc.special_tokens_mask
        content = [i for i, m in zip(enc.ids, mask) if m == 0]
        total_tokens += len(content)
        unk += sum(1 for i in content if i == unk_id)
        used_ids.update(enc.ids)
        total_words += len(line.split())
        decoded = tok.decode(enc.ids, skip_special_tokens=True)
        if decoded == line:
            rt_exact += 1
        if _norm_ws(decoded) == _norm_ws(line):
            rt_norm += 1

    # Reserved-token integrity.
    reserved = {t: tok.token_to_id(t) for t in EXPECTED_SPECIAL_IDS}
    reserved_ok = reserved == EXPECTED_SPECIAL_IDS

    denom_t = max(1, total_tokens)
    denom_s = max(1, sentences)
    denom_w = max(1, total_words)
    return {
        "tokenizer_dir": os.path.relpath(tokenizer_dir),
        "eval_files": [os.path.relpath(f) for f in eval_files],
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vocab_size": vocab_size,
        "sentences": sentences,
        "content_tokens": total_tokens,
        "words": total_words,
        "unknown_token_rate": unk / denom_t,
        "unknown_tokens": unk,
        "avg_tokens_per_sentence": total_tokens / denom_s,
        "avg_tokens_per_word": total_tokens / denom_w,
        "roundtrip_exact_rate": rt_exact / denom_s,
        "roundtrip_normalized_rate": rt_norm / denom_s,
        "vocabulary_utilization": len(used_ids) / max(1, vocab_size),
        "unique_ids_used": len(used_ids),
        "reserved_token_integrity": reserved_ok,
        "reserved_token_ids": reserved,
    }


def write_reports(metrics: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "evaluation.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    mpath = os.path.join(out_dir, "evaluation.md")
    m = metrics
    lines = [
        "# Tokenizer evaluation", "",
        f"_Generated: {m['created_utc']}_  ·  tokenizer: `{m['tokenizer_dir']}`", "",
        "## Metrics", "",
        "| metric | value |", "|---|---:|",
        f"| vocab size | {m['vocab_size']:,} |",
        f"| sentences evaluated | {m['sentences']:,} |",
        f"| content tokens | {m['content_tokens']:,} |",
        f"| words | {m['words']:,} |",
        f"| unknown-token rate | {m['unknown_token_rate']:.4%} |",
        f"| avg tokens / sentence | {m['avg_tokens_per_sentence']:.3f} |",
        f"| avg tokens / word | {m['avg_tokens_per_word']:.3f} |",
        f"| round-trip exact | {m['roundtrip_exact_rate']:.2%} |",
        f"| round-trip (whitespace-normalized) | {m['roundtrip_normalized_rate']:.2%} |",
        f"| vocabulary utilization | {m['vocabulary_utilization']:.2%} "
        f"({m['unique_ids_used']:,}/{m['vocab_size']:,}) |",
        f"| reserved-token integrity | {'OK' if m['reserved_token_integrity'] else 'FAIL'} |",
        "",
        f"Reserved token ids: `{m['reserved_token_ids']}`", "",
        "> Intrinsic metrics only. Round-trip exact rate is naturally low for subword "
        "tokenizers (spacing/casing may change); the whitespace-normalized rate is the "
        "meaningful fidelity signal. Vocabulary utilization is low on tiny eval corpora.",
    ]
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return jpath, mpath
