"""Offline tokenize-and-pack pipeline + memory-mapped packed-token dataset (real-text stage).

Article-level truncation (tokenize each row, keep only ``max_seq_length`` tokens) discards most
of every long Wikipedia article and materializes the whole encoded corpus as Python lists. This
module instead **packs** a tokenized corpus into fixed-length rows on disk:

  data/tokenized/<run>/
    manifest.json
    train/shard-00000.npy ...        int array [num_sequences, sequence_length]
    validation/shard-00000.npy ...

Each row is ``[CLS] content... [SEP] PAD...`` — token IDs only. **No MLM masks or labels are
stored**; masking stays dynamic in the collator. Long documents are split across as many
sequences as needed; a sequence never crosses a document boundary (v1). Only the final chunk of
a document is padded. dtype is uint16 when vocab_size <= 65535, else uint32.

Dependency-light: numpy + ``tokenizers`` + stdlib. No torch import at module load (the dataset
class imports torch lazily) so packing tooling stays importable without torch.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FORMAT_VERSION = 1
EXPECTED_SPECIAL_TOKENS = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4}
PACKING_POLICY = (
    "v1: one input line = one document; content encoded WITHOUT auto special tokens, then "
    "chunked to (sequence_length - 2) tokens; each chunk framed as [CLS] content [SEP]; only "
    "the final chunk of a document is padded; sequences never cross a document boundary; "
    "documents processed in sorted-file, in-file order (deterministic)."
)


def dtype_for_vocab(vocab_size: int) -> str:
    return "uint16" if vocab_size <= 65535 else "uint32"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit(root: Optional[str]) -> Optional[str]:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=root or os.getcwd())
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def resolve_txt_files(inputs) -> list[str]:
    """Recursively resolve a list of files/dirs into a sorted list of .txt files."""
    files: list[str] = []
    for item in inputs or []:
        if os.path.isdir(item):
            for root, _d, names in os.walk(item):
                for n in sorted(names):
                    if n.lower().endswith(".txt"):
                        files.append(os.path.join(root, n))
        elif os.path.isfile(item) and item.lower().endswith(".txt"):
            files.append(item)
    return sorted(set(files))


def _detect_language(path: str, sample_lines: int = 50) -> str:
    try:
        from .corpus import document_language
        buf = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    buf.append(line.strip())
                if len(buf) >= sample_lines:
                    break
        return document_language(" ".join(buf)) if buf else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------------------- #
@dataclass
class PackCounters:
    documents_read: int = 0
    documents_skipped: int = 0
    source_content_tokens: int = 0
    packed_sequences: int = 0
    padding_tokens: int = 0
    unknown_tokens: int = 0

    def as_dict(self, sequence_length: int) -> dict:
        total_slots = self.packed_sequences * sequence_length
        content_slots = total_slots - self.padding_tokens
        return {
            "documents_read": self.documents_read,
            "documents_skipped": self.documents_skipped,
            "source_content_tokens": self.source_content_tokens,
            "packed_sequences": self.packed_sequences,
            "padding_tokens": self.padding_tokens,
            "content_slots": content_slots,
            "packing_efficiency": (content_slots / total_slots) if total_slots else 0.0,
            "unknown_tokens": self.unknown_tokens,
            "unknown_token_rate": (self.unknown_tokens / self.source_content_tokens)
            if self.source_content_tokens else 0.0,
        }


# --------------------------------------------------------------------------------------- #
# Shard writing
# --------------------------------------------------------------------------------------- #
class _ShardWriter:
    """Accumulates rows and flushes fixed-size shards atomically (tmp file + os.replace)."""

    def __init__(self, out_dir: str, prefix: str, seq_len: int, np_dtype, per_shard: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.seq_len = seq_len
        self.np_dtype = np_dtype
        self.per_shard = per_shard
        os.makedirs(out_dir, exist_ok=True)
        self._buf: list[np.ndarray] = []
        self._index = 0
        self.shards: list[dict] = []

    def add(self, row: np.ndarray) -> None:
        self._buf.append(row)
        if len(self._buf) >= self.per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        arr = np.stack(self._buf).astype(self.np_dtype, copy=False)
        name = f"{self.prefix}-{self._index:05d}.npy"
        final = os.path.join(self.out_dir, name)
        tmp = final + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, final)  # atomic publish
        self.shards.append({"name": name, "shape": list(arr.shape),
                            "bytes": os.path.getsize(final), "sha256": sha256_file(final)})
        self._index += 1
        self._buf = []

    @property
    def n_sequences(self) -> int:
        return sum(s["shape"][0] for s in self.shards)


# --------------------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------------------- #
def _load_tokenizer(tokenizer):
    """Accept a path (dir or tokenizer.json) or an already-loaded ``tokenizers.Tokenizer``."""
    from tokenizers import Tokenizer
    if not isinstance(tokenizer, str):
        return tokenizer, None  # already a Tokenizer object
    path = tokenizer
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer.json not found: {path}")
    return Tokenizer.from_file(path), path


def _verify_tokenizer(tok, expected_vocab_size: int) -> None:
    vs = tok.get_vocab_size()
    if vs != expected_vocab_size:
        raise ValueError(f"tokenizer vocab_size {vs} != expected {expected_vocab_size} "
                         "(refusing to pack with a mismatched tokenizer)")
    for t, i in EXPECTED_SPECIAL_TOKENS.items():
        got = tok.token_to_id(t)
        if got != i:
            raise ValueError(f"special token {t} has id {got}, expected {i} "
                             "(reserved-token integrity failed)")


def _pack_split(files: list[str], writer: _ShardWriter, tok, seq_len: int,
                counters: PackCounters, progress_every: int, split: str) -> list[dict]:
    cls, sep, pad = 1, 2, 0
    unk = EXPECTED_SPECIAL_TOKENS["[UNK]"]
    content_per_seq = seq_len - 2
    if content_per_seq < 1:
        raise ValueError("sequence_length must be >= 3")
    source_files = []
    for path in files:
        n_docs = 0
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                doc = line.strip()
                counters.documents_read += 1
                if not doc:
                    counters.documents_skipped += 1
                    continue
                ids = tok.encode(doc, add_special_tokens=False).ids
                if not ids:
                    counters.documents_skipped += 1
                    continue
                n_docs += 1
                counters.source_content_tokens += len(ids)
                counters.unknown_tokens += sum(1 for i in ids if i == unk)
                # Chunk this document; never cross into the next document.
                for start in range(0, len(ids), content_per_seq):
                    chunk = ids[start:start + content_per_seq]
                    row = np.full(seq_len, pad, dtype=np.int64)
                    row[0] = cls
                    row[1:1 + len(chunk)] = chunk
                    row[1 + len(chunk)] = sep
                    counters.packed_sequences += 1
                    counters.padding_tokens += seq_len - (len(chunk) + 2)
                    writer.add(row)
                if counters.documents_read % progress_every == 0:
                    print(f"[pack:{split}] read {counters.documents_read:,} docs, "
                          f"{writer.n_sequences + len(writer._buf):,} sequences so far")
        source_files.append({"path": os.path.relpath(path), "sha256": sha256_file(path),
                             "language": _detect_language(path), "split": split,
                             "documents": n_docs})
    writer.flush()
    return source_files


def pack_corpus(tokenizer, train_inputs, validation_inputs, output_dir: str,
                sequence_length: int = 128, sequences_per_shard: int = 100_000,
                seed: int = 42, overwrite: bool = False, expected_vocab_size: int = 32000,
                progress_every: int = 10_000, project_root: Optional[str] = None) -> dict:
    """Tokenize + pack a corpus to ``output_dir``. Returns the manifest dict. Raises loudly."""
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"output dir '{output_dir}' exists. Use --overwrite for a clean rebuild "
                "(restart is only supported via clean overwrite, not partial mixed output).")
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tok, tok_path = _load_tokenizer(tokenizer)
    _verify_tokenizer(tok, expected_vocab_size)

    np_dtype = np.uint16 if expected_vocab_size <= 65535 else np.uint32
    train_files = resolve_txt_files(train_inputs)
    val_files = resolve_txt_files(validation_inputs)
    if not train_files:
        raise ValueError("no .txt training files resolved from --train-input")

    counters = PackCounters()
    tw = _ShardWriter(os.path.join(output_dir, "train"), "shard", sequence_length,
                      np_dtype, sequences_per_shard)
    train_sources = _pack_split(train_files, tw, tok, sequence_length, counters,
                                progress_every, "train")
    val_sources: list[dict] = []
    vw = None
    if val_files:
        vw = _ShardWriter(os.path.join(output_dir, "validation"), "shard", sequence_length,
                          np_dtype, sequences_per_shard)
        val_sources = _pack_split(val_files, vw, tok, sequence_length, counters,
                                  progress_every, "validation")

    if tw.n_sequences == 0:
        raise ValueError("no training sequences were produced (empty corpus?)")

    tok_sha = sha256_file(tok_path) if tok_path and os.path.exists(tok_path) else None
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_utc": _utc_now(),
        "git_commit": _git_commit(project_root),
        "seed": seed,
        "sequence_length": sequence_length,
        "dtype": dtype_for_vocab(expected_vocab_size),
        "tokenizer_path": (os.path.relpath(tok_path) if tok_path else None),
        "tokenizer_sha256": tok_sha,
        "tokenizer_vocab_size": tok.get_vocab_size(),
        "special_token_ids": dict(EXPECTED_SPECIAL_TOKENS),
        "packing_policy": PACKING_POLICY,
        "sequences_per_shard": sequences_per_shard,
        "train_sequences": tw.n_sequences,
        "validation_sequences": (vw.n_sequences if vw else 0),
        "source_files": train_sources + val_sources,
        "shards": {"train": tw.shards, "validation": (vw.shards if vw else [])},
        "counters": counters.as_dict(sequence_length),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


# --------------------------------------------------------------------------------------- #
# Memory-mapped dataset
# --------------------------------------------------------------------------------------- #
@dataclass
class _ShardHandle:
    path: str
    n_rows: int
    array: object = field(default=None)


class PackedTokenDataset:
    """Memory-mapped dataset over packed .npy shards for one split (train / validation).

    Rows are memory-mapped; ``__getitem__`` copies a single row (bounded memory). Returns a
    1-D torch.LongTensor of length ``sequence_length``.
    """

    def __init__(self, manifest_dir: str, split: str) -> None:
        manifest_dir = os.path.abspath(manifest_dir)
        with open(os.path.join(manifest_dir, "manifest.json"), "r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")
        self.split = split
        self.sequence_length = int(self.manifest["sequence_length"])
        self._handles: list[_ShardHandle] = []
        self._cum: list[int] = []           # cumulative start row of each shard
        total = 0
        for s in self.manifest["shards"].get(split, []):
            path = os.path.join(manifest_dir, split, s["name"])
            n = int(s["shape"][0])
            self._handles.append(_ShardHandle(path=path, n_rows=n))
            self._cum.append(total)
            total += n
        self._length = total

    def __len__(self) -> int:
        return self._length

    def _shard_array(self, idx: int):
        h = self._handles[idx]
        if h.array is None:
            h.array = np.load(h.path, mmap_mode="r")  # memory-mapped, lazy
        return h.array

    def __getitem__(self, index: int):
        import torch
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        # Map global row -> (shard, local row).
        shard = bisect.bisect_right(self._cum, index) - 1
        local = index - self._cum[shard]
        row = np.asarray(self._shard_array(shard)[local], dtype=np.int64)
        return torch.from_numpy(row)
