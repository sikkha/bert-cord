"""Corpus preparation for tokenizer training (Tokenizer Milestone).

Reads text from ``.txt`` / ``.md`` / ``.jsonl`` files (and, optionally, a Hugging Face dataset),
normalizes Unicode, drops empties, exact-deduplicates, deterministically shuffles, computes
per-script language statistics, and writes a sharded corpus plus a manifest and a Markdown
report. The goal is a **reproducible engineering pipeline**, not maximal quality.

Everything here is dependency-light (stdlib + optional ``datasets``); nothing imports torch.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Optional

_NORMALIZATION_FORMS = {"NFC", "NFKC", "NFD", "NFKD", "none"}

# Unicode script ranges used for lightweight language statistics.
_SCRIPT_RANGES = [
    ("thai", 0x0E00, 0x0E7F),
    ("latin_basic", 0x0041, 0x024F),
    ("cyrillic", 0x0400, 0x04FF),
    ("arabic", 0x0600, 0x06FF),
    ("devanagari", 0x0900, 0x097F),
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("hangul", 0xAC00, 0xD7A3),
    ("cjk", 0x4E00, 0x9FFF),
]


@dataclass
class CorpusConfig:
    normalization: str = "NFC"       # NFC | NFKC | NFD | NFKD | none
    dedup: bool = True               # exact (hash) deduplication
    min_chars: int = 1               # drop documents shorter than this (after strip)
    shuffle_seed: int = 42
    val_fraction: float = 0.02       # held-out validation split
    shard_size: int = 100_000        # documents per train shard
    text_field: str = "text"         # for jsonl / HF datasets

    def __post_init__(self) -> None:
        if self.normalization not in _NORMALIZATION_FORMS:
            raise ValueError(f"normalization must be one of {_NORMALIZATION_FORMS}, "
                             f"got {self.normalization!r}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")


# --------------------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------------------- #
def _normalize(text: str, form: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if form != "none":
        text = unicodedata.normalize(form, text)
    return text


def iter_txt(path: str) -> Iterator[str]:
    """Yield each non-empty line of a .txt / .md file as one document."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                yield line


def iter_jsonl(path: str, text_field: str) -> Iterator[str]:
    """Yield ``record[text_field]`` for each JSON line."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = obj.get(text_field) if isinstance(obj, dict) else None
            if isinstance(val, str) and val.strip():
                yield val


def iter_file(path: str, text_field: str = "text") -> Iterator[str]:
    """Dispatch on file extension (.txt/.md/.markdown -> lines; .jsonl/.ndjson -> field)."""
    lower = path.lower()
    if lower.endswith((".jsonl", ".ndjson")):
        yield from iter_jsonl(path, text_field)
    else:  # .txt, .md, .markdown, and anything else treated as plain text
        yield from iter_txt(path)


def expand_inputs(inputs: Iterable[str]) -> list[str]:
    """Expand a list of files/dirs into a sorted list of files (recurses into dirs)."""
    files: list[str] = []
    exts = (".txt", ".md", ".markdown", ".jsonl", ".ndjson")
    for item in inputs:
        if os.path.isdir(item):
            for root, _d, names in os.walk(item):
                for n in sorted(names):
                    if n.lower().endswith(exts):
                        files.append(os.path.join(root, n))
        elif os.path.isfile(item):
            files.append(item)
    return sorted(set(files))


def iter_hf_dataset(name: str, config: Optional[str], split: str, text_field: str,
                    max_docs: Optional[int] = None) -> Iterator[str]:
    """Stream documents from a Hugging Face dataset (requires ``datasets``)."""
    from datasets import load_dataset  # local import: optional dependency

    ds = load_dataset(name, config, split=split, streaming=True)
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        val = ex.get(text_field) if isinstance(ex, dict) else None
        if isinstance(val, str) and val.strip():
            yield val


# --------------------------------------------------------------------------------------- #
# Language statistics
# --------------------------------------------------------------------------------------- #
def script_of_char(ch: str) -> str:
    o = ord(ch)
    if ch.isdigit():
        return "digit"
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= o <= hi:
            return name
    if ch.isspace():
        return "space"
    if not ch.isalnum():
        return "punct_symbol"
    return "other"


def document_language(text: str) -> str:
    """Dominant alphabetic script of a document (ignores spaces/punct/digits)."""
    counts: dict[str, int] = {}
    for ch in text:
        s = script_of_char(ch)
        if s in ("space", "punct_symbol", "digit", "other"):
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "unknown"
    dom = max(counts, key=counts.get)
    return {"latin_basic": "latin"}.get(dom, dom)


# --------------------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------------------- #
@dataclass
class CorpusStats:
    documents_read: int = 0
    documents_kept: int = 0
    duplicates_removed: int = 0
    empty_removed: int = 0
    total_chars: int = 0
    per_language_docs: dict = field(default_factory=dict)
    per_language_chars: dict = field(default_factory=dict)


def prepare_corpus(inputs: Iterable[str], output_dir: str, cfg: CorpusConfig,
                   hf_dataset: Optional[dict] = None,
                   source_labels: Optional[dict] = None) -> dict:
    """Read -> normalize -> filter -> dedup -> shuffle -> shard; write manifest + report.

    ``hf_dataset`` (optional): {"name","config","split","max_docs"}. ``source_labels`` maps a
    source path/name to a language label to override auto-detection (optional).
    Returns the manifest dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = expand_inputs(inputs)

    stats = CorpusStats()
    seen: set = set()
    kept: list[str] = []
    sources: list[dict] = []

    def _consume(source_name: str, it: Iterator[str]) -> None:
        n_read = n_kept = 0
        for raw in it:
            n_read += 1
            stats.documents_read += 1
            text = _normalize(raw, cfg.normalization)
            if len(text) < cfg.min_chars:
                stats.empty_removed += 1
                continue
            if cfg.dedup:
                h = hashlib.sha1(text.encode("utf-8")).digest()
                if h in seen:
                    stats.duplicates_removed += 1
                    continue
                seen.add(h)
            kept.append(text)
            n_kept += 1
        sources.append({"source": source_name, "documents_read": n_read,
                        "documents_kept": n_kept})

    for path in files:
        _consume(os.path.relpath(path), iter_file(path, cfg.text_field))
    if hf_dataset:
        label = f"hf:{hf_dataset['name']}:{hf_dataset.get('config')}:{hf_dataset.get('split')}"
        _consume(label, iter_hf_dataset(
            hf_dataset["name"], hf_dataset.get("config"), hf_dataset.get("split", "train"),
            cfg.text_field, hf_dataset.get("max_docs")))

    # Language statistics on kept docs.
    for text in kept:
        lang = (source_labels or {}).get("*", None) or document_language(text)
        stats.per_language_docs[lang] = stats.per_language_docs.get(lang, 0) + 1
        stats.per_language_chars[lang] = stats.per_language_chars.get(lang, 0) + len(text)
        stats.total_chars += len(text)
    stats.documents_kept = len(kept)

    # Deterministic shuffle.
    rng = random.Random(cfg.shuffle_seed)
    rng.shuffle(kept)

    # Validation split (deterministic: take the tail after shuffle).
    n_val = int(round(len(kept) * cfg.val_fraction))
    val_docs = kept[len(kept) - n_val:] if n_val > 0 else []
    train_docs = kept[: len(kept) - n_val] if n_val > 0 else kept

    # Write shards.
    shard_files = _write_shards(train_docs, output_dir, cfg.shard_size)
    val_file = None
    if val_docs:
        val_file = os.path.join(output_dir, "validation.txt")
        _write_lines(val_docs, val_file)

    manifest = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": asdict(cfg),
        "sources": sources,
        "hf_dataset": hf_dataset,
        "statistics": asdict(stats),
        "train_shards": [{"path": os.path.basename(p), "documents": _count_lines(p),
                          "sha256": _sha256(p), "bytes": os.path.getsize(p)}
                         for p in shard_files],
        "validation": (None if not val_file else
                       {"path": os.path.basename(val_file), "documents": len(val_docs),
                        "sha256": _sha256(val_file), "bytes": os.path.getsize(val_file)}),
        "train_documents": len(train_docs),
        "validation_documents": len(val_docs),
    }
    with open(os.path.join(output_dir, "corpus_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    _write_report(manifest, os.path.join(output_dir, "corpus_report.md"))
    return manifest


def _write_shards(docs: list[str], out_dir: str, shard_size: int) -> list[str]:
    paths = []
    if not docs:
        # Always emit at least one (possibly empty) shard for a stable layout.
        p = os.path.join(out_dir, "train-00000.txt")
        _write_lines([], p)
        return [p]
    for i in range(0, len(docs), shard_size):
        idx = i // shard_size
        p = os.path.join(out_dir, f"train-{idx:05d}.txt")
        _write_lines(docs[i:i + shard_size], p)
        paths.append(p)
    return paths


def _write_lines(docs: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(d.replace("\n", " ") + "\n")


def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _write_report(manifest: dict, path: str) -> None:
    st = manifest["statistics"]
    lines = ["# Tokenizer corpus report", "",
             f"_Generated: {manifest['created_utc']}_", "",
             "## Summary", "",
             f"- Documents read: **{st['documents_read']:,}**",
             f"- Documents kept: **{st['documents_kept']:,}**",
             f"- Empty/short removed: {st['empty_removed']:,}",
             f"- Exact duplicates removed: {st['duplicates_removed']:,}",
             f"- Total characters: {st['total_chars']:,}",
             f"- Train documents: {manifest['train_documents']:,} in "
             f"{len(manifest['train_shards'])} shard(s)",
             f"- Validation documents: {manifest['validation_documents']:,}",
             f"- Normalization: `{manifest['config']['normalization']}` · "
             f"dedup: {manifest['config']['dedup']} · shuffle_seed: "
             f"{manifest['config']['shuffle_seed']}", "",
             "## Language distribution (dominant script per document)", "",
             "| language | documents | characters |", "|---|---:|---:|"]
    langs = sorted(st["per_language_docs"], key=lambda k: -st["per_language_docs"][k])
    for lang in langs:
        lines.append(f"| {lang} | {st['per_language_docs'][lang]:,} | "
                     f"{st['per_language_chars'].get(lang, 0):,} |")
    lines += ["", "## Sources", "", "| source | read | kept |", "|---|---:|---:|"]
    for s in manifest["sources"]:
        lines.append(f"| `{s['source']}` | {s['documents_read']:,} | {s['documents_kept']:,} |")
    lines += ["", "> Corpus outputs are git-ignored. This is a reproducible engineering "
              "pipeline for tokenizer training, not a quality-optimized corpus."]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
