#!/usr/bin/env python3
"""Validate a packed token corpus (offline; no network).

Checks the manifest schema, shard existence + SHA-256, dtype/dimensions, id bounds, framing
([CLS] first, [SEP] before padding, [PAD] only after [SEP]), the absence of stored [MASK] or
MLM labels, non-empty train/validation splits, and (optionally) the tokenizer checksum.

Usage:
  python scripts/validate_packed_corpus.py data/tokenized/bert_cord_en_th_128_v1 \
    [--tokenizer artifacts/tokenizers/bert-cord-byte-bpe-32k] [--require-validation]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

PAD, CLS, SEP, MASK = 0, 1, 2, 3
REQUIRED_MANIFEST_KEYS = [
    "format_version", "sequence_length", "dtype", "tokenizer_path", "tokenizer_sha256",
    "tokenizer_vocab_size", "special_token_ids", "packing_policy", "train_sequences",
    "validation_sequences", "shards", "counters", "source_files", "git_commit", "seed",
]
_DTYPE = {"uint16": np.uint16, "uint32": np.uint32}


class _R:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def ok(self):
        return all(ok for _, ok, _ in self.checks)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _check_split(r: _R, base: str, split: str, manifest: dict, np_dtype, seq_len: int,
                 vocab: int) -> int:
    shards = manifest["shards"].get(split, [])
    n_rows = 0
    all_ok_dtype = True
    all_ok_dims = True
    all_ok_bounds = True
    cls_ok = True
    sep_frame_ok = True
    no_mask = True
    checksum_ok = True
    for s in shards:
        path = os.path.join(base, split, s["name"])
        if not os.path.exists(path):
            r.add(f"{split}: shard exists ({s['name']})", False, "missing")
            continue
        if _sha256(path) != s["sha256"] or os.path.getsize(path) != s["bytes"]:
            checksum_ok = False
        arr = np.load(path, mmap_mode="r")
        all_ok_dtype = all_ok_dtype and (arr.dtype == np_dtype)
        all_ok_dims = all_ok_dims and (arr.ndim == 2 and arr.shape[1] == seq_len
                                       and list(arr.shape) == s["shape"])
        n_rows += arr.shape[0]
        if arr.shape[0] == 0:
            continue
        a = np.asarray(arr)  # read shard (bounded by sequences_per_shard)
        all_ok_bounds = all_ok_bounds and bool((a >= 0).all() and (a < vocab).all())
        cls_ok = cls_ok and bool((a[:, 0] == CLS).all())
        no_mask = no_mask and bool(not (a == MASK).any())
        # Framing: every row has a SEP; PAD only after the first SEP; no PAD at/before SEP.
        has_sep = (a == SEP).any(axis=1)
        if not bool(has_sep.all()):
            sep_frame_ok = False
        else:
            sep_idx = (a == SEP).argmax(axis=1)                 # first SEP per row
            cols = np.arange(seq_len)[None, :]
            after = cols > sep_idx[:, None]
            is_pad = (a == PAD)
            after_all_pad = np.where(after, is_pad, True).all()
            before_no_pad = np.where(~after, ~is_pad, True).all()
            sep_frame_ok = sep_frame_ok and bool(after_all_pad and before_no_pad)

    r.add(f"{split}: shard checksums match", checksum_ok)
    r.add(f"{split}: dtype == {manifest['dtype']}", all_ok_dtype)
    r.add(f"{split}: dimensions [*, {seq_len}] match manifest", all_ok_dims)
    r.add(f"{split}: all ids in [0, {vocab})", all_ok_bounds)
    r.add(f"{split}: every row begins with [CLS]", cls_ok)
    r.add(f"{split}: [SEP] before padding & [PAD] only after [SEP]", sep_frame_ok)
    r.add(f"{split}: no [MASK] tokens stored", no_mask)
    r.add(f"{split}: sequence count matches manifest ({n_rows})",
          n_rows == manifest[f"{split}_sequences"])
    return n_rows


def validate(base: str, tokenizer: str | None, require_validation: bool) -> _R:
    r = _R()
    base = os.path.abspath(base)
    mpath = os.path.join(base, "manifest.json")
    if not os.path.exists(mpath):
        r.add("manifest.json exists", False, mpath)
        return r
    manifest = json.load(open(mpath, encoding="utf-8"))

    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    r.add("manifest schema (required keys)", not missing,
          f"missing: {missing}" if missing else "")
    # No stored MLM labels declared anywhere.
    r.add("no MLM labels in format", "labels" not in manifest and not any(
        "label" in s["name"].lower() for sp in manifest.get("shards", {}).values() for s in sp))

    seq_len = int(manifest["sequence_length"])
    vocab = int(manifest["tokenizer_vocab_size"])
    np_dtype = _DTYPE.get(manifest["dtype"])
    r.add("dtype is uint16/uint32", np_dtype is not None, manifest["dtype"])
    r.add("special_token_ids canonical",
          manifest.get("special_token_ids") == {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2,
                                                "[MASK]": 3, "[UNK]": 4})

    n_train = _check_split(r, base, "train", manifest, np_dtype, seq_len, vocab)
    n_val = _check_split(r, base, "validation", manifest, np_dtype, seq_len, vocab)
    r.add("train split non-empty", n_train > 0)
    if require_validation:
        r.add("validation split non-empty", n_val > 0)
    else:
        r.add("validation split present", True, f"{n_val} sequences")

    # Tokenizer checksum.
    if tokenizer:
        tok_json = os.path.join(tokenizer, "tokenizer.json") if os.path.isdir(tokenizer) \
            else tokenizer
        if os.path.exists(tok_json):
            r.add("tokenizer checksum matches manifest",
                  _sha256(tok_json) == manifest.get("tokenizer_sha256"),
                  f"file={_sha256(tok_json)[:12]}… manifest={str(manifest.get('tokenizer_sha256'))[:12]}…")
        else:
            r.add("tokenizer checksum matches manifest", False, f"not found: {tok_json}")
    else:
        r.add("tokenizer checksum matches manifest", True,
              "skipped (no --tokenizer given)")
    return r


def main() -> int:
    p = argparse.ArgumentParser(description="Validate a packed token corpus (offline).")
    p.add_argument("packed_dir")
    p.add_argument("--tokenizer", default=None, help="Tokenizer dir/json to checksum-verify.")
    p.add_argument("--require-validation", action="store_true")
    args = p.parse_args()
    if not os.path.isdir(args.packed_dir):
        print(f"[validate_packed] not a directory: {args.packed_dir}", file=sys.stderr)
        return 2
    print("=" * 68)
    print(f"[validate_packed] {args.packed_dir}")
    print("-" * 68)
    r = validate(args.packed_dir, args.tokenizer, args.require_validation)
    n_ok = sum(1 for _, ok, _ in r.checks if ok)
    print("-" * 68)
    print(f"[validate_packed] {n_ok}/{len(r.checks)} checks passed -> "
          f"{'PASS' if r.ok else 'FAIL'}")
    print("=" * 68)
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
