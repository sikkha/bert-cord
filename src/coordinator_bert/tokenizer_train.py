"""Config-driven tokenizer training for BERT-Cord (Tokenizer Milestone).

Trains a Hugging Face ``tokenizers`` tokenizer using one of three subword algorithms
(byte-level BPE, Unigram, WordPiece) from a YAML config, and writes a self-contained artifact
directory. Special tokens are pinned to fixed ids matching ``coordinator_bert.data.SpecialTokens``
([PAD]=0, [CLS]=1, [SEP]=2, [MASK]=3, [UNK]=4) so the tokenizer stays compatible with the MLM
pipeline once frozen.

The goal is a robust, reproducible engineering pipeline — not maximal tokenizer quality yet.
This module has no torch dependency; ``tokenizers``/``pyyaml`` are required (the ``train`` extra).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Optional

# Fixed special tokens + ids (must match coordinator_bert.data.SpecialTokens).
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
SPECIAL_IDS = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4}
ALGORITHMS = ("byte_bpe", "unigram", "wordpiece")


@dataclass
class TokenizerTrainConfig:
    name: str = "bert-cord-wordpiece-32k"
    algorithm: str = "wordpiece"          # byte_bpe | unigram | wordpiece
    vocab_size: int = 32000
    min_frequency: int = 2
    normalization: str = "NFC"            # NFC | NFKC | NFD | NFKD | none
    lowercase: bool = False
    byte_fallback: bool = True            # unigram/bpe only (ignored for byte_bpe/wordpiece)
    add_bert_postprocessor: bool = True    # add [CLS] .. [SEP] template
    model_max_length: int = 512
    special_tokens: list = field(default_factory=lambda: list(SPECIAL_TOKENS))

    @classmethod
    def from_dict(cls, d: dict) -> "TokenizerTrainConfig":
        from dataclasses import fields
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in (d or {}).items() if k in known})
        if cfg.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {cfg.algorithm!r}")
        # Special tokens must start with the canonical five in the canonical order.
        if list(cfg.special_tokens[:5]) != SPECIAL_TOKENS:
            raise ValueError("special_tokens must begin with "
                             f"{SPECIAL_TOKENS} (fixed ids 0..4)")
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "TokenizerTrainConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        # Accept either a flat dict or a {tokenizer: {...}} wrapper.
        d = raw.get("tokenizer", raw)
        return cls.from_dict(d)


# --------------------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------------------- #
def _build_normalizer(cfg: TokenizerTrainConfig):
    from tokenizers import normalizers
    seq = []
    form = cfg.normalization
    if form != "none":
        seq.append({"NFC": normalizers.NFC, "NFKC": normalizers.NFKC,
                    "NFD": normalizers.NFD, "NFKD": normalizers.NFKD}[form]())
    if cfg.lowercase:
        seq.append(normalizers.Lowercase())
    if not seq:
        return None
    return seq[0] if len(seq) == 1 else normalizers.Sequence(seq)


def build_tokenizer(cfg: TokenizerTrainConfig):
    """Construct an untrained Tokenizer + its Trainer for the configured algorithm."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    algo = cfg.algorithm
    if algo == "byte_bpe":
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        norm = _build_normalizer(cfg)
        if norm is not None:
            tok.normalizer = norm
        trainer = trainers.BpeTrainer(
            vocab_size=cfg.vocab_size, min_frequency=cfg.min_frequency,
            special_tokens=cfg.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
    elif algo == "unigram":
        tok = Tokenizer(models.Unigram())
        norm = _build_normalizer(cfg)
        if norm is not None:
            tok.normalizer = norm
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        try:
            from tokenizers import decoders as _dec
            tok.decoder = _dec.Metaspace()
        except Exception:  # noqa: BLE001
            pass
        kwargs = dict(vocab_size=cfg.vocab_size, special_tokens=cfg.special_tokens,
                      unk_token="[UNK]")
        try:
            trainer = trainers.UnigramTrainer(byte_fallback=cfg.byte_fallback, **kwargs)
        except TypeError:
            trainer = trainers.UnigramTrainer(**kwargs)  # older tokenizers w/o byte_fallback
    else:  # wordpiece
        from tokenizers.normalizers import BertNormalizer
        from tokenizers.pre_tokenizers import BertPreTokenizer
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=100))
        tok.normalizer = BertNormalizer(lowercase=cfg.lowercase)
        tok.pre_tokenizer = BertPreTokenizer()
        tok.decoder = decoders.WordPiece(prefix="##")
        trainer = trainers.WordPieceTrainer(
            vocab_size=cfg.vocab_size, min_frequency=cfg.min_frequency,
            special_tokens=cfg.special_tokens,
        )
    return tok, trainer


def _add_postprocessor(tok) -> None:
    from tokenizers.processors import TemplateProcessing
    cls_id, sep_id = SPECIAL_IDS["[CLS]"], SPECIAL_IDS["[SEP]"]
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )


# --------------------------------------------------------------------------------------- #
# Training + artifact writing
# --------------------------------------------------------------------------------------- #
def train_tokenizer(cfg: TokenizerTrainConfig, input_files: list[str], output_dir: str,
                    corpus_manifest: Optional[str] = None,
                    project_root: Optional[str] = None) -> dict:
    """Train the tokenizer and write the artifact directory. Returns the manifest dict."""
    if not input_files:
        raise ValueError("no input files provided for tokenizer training")
    for f in input_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"training input not found: {f}")

    tok, trainer = build_tokenizer(cfg)
    tok.train(input_files, trainer)

    # Verify + pin special-token ids BEFORE post-processing.
    for t, expected in SPECIAL_IDS.items():
        got = tok.token_to_id(t)
        if got != expected:
            raise RuntimeError(f"special token {t} got id {got}, expected {expected} "
                               "(reserved-token integrity failed)")
    if cfg.add_bert_postprocessor:
        _add_postprocessor(tok)

    os.makedirs(output_dir, exist_ok=True)
    tok_path = os.path.join(output_dir, "tokenizer.json")
    tok.save(tok_path)

    _write_tokenizer_config(cfg, output_dir)
    _write_special_tokens_map(output_dir)

    manifest = _build_manifest(cfg, tok, tok_path, input_files, corpus_manifest, project_root)
    with open(os.path.join(output_dir, "tokenizer_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    _write_readme(cfg, tok, manifest, output_dir)
    return manifest


def _write_tokenizer_config(cfg: TokenizerTrainConfig, out: str) -> None:
    conf = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": cfg.model_max_length,
        "do_lower_case": cfg.lowercase,
        "clean_up_tokenization_spaces": True,
        "pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]",
        "mask_token": "[MASK]", "unk_token": "[UNK]",
        "algorithm": cfg.algorithm, "normalization": cfg.normalization,
        "byte_fallback": cfg.byte_fallback,
    }
    with open(os.path.join(out, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(conf, fh, indent=2)


def _write_special_tokens_map(out: str) -> None:
    m = {"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]",
         "mask_token": "[MASK]", "unk_token": "[UNK]"}
    with open(os.path.join(out, "special_tokens_map.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)


def _git_commit(root: Optional[str]) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=root or os.getcwd())
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _build_manifest(cfg, tok, tok_path, input_files, corpus_manifest, project_root) -> dict:
    corpus_ref = None
    if corpus_manifest and os.path.exists(corpus_manifest):
        corpus_ref = {"path": os.path.relpath(corpus_manifest),
                      "sha256": _sha256(corpus_manifest)}
    return {
        "name": cfg.name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "algorithm": cfg.algorithm,
        "requested_vocab_size": cfg.vocab_size,
        "actual_vocab_size": tok.get_vocab_size(),
        "normalization": cfg.normalization,
        "lowercase": cfg.lowercase,
        "byte_fallback": cfg.byte_fallback,
        "add_bert_postprocessor": cfg.add_bert_postprocessor,
        "special_tokens": cfg.special_tokens,
        "special_token_ids": {t: tok.token_to_id(t) for t in cfg.special_tokens},
        "training_inputs": [os.path.relpath(f) for f in input_files],
        "corpus_manifest": corpus_ref,
        "git_commit": _git_commit(project_root),
        "tokenizer_json_sha256": _sha256(tok_path),
        "tokenizer_json_bytes": os.path.getsize(tok_path),
        "config": asdict(cfg),
    }


def _write_readme(cfg, tok, manifest, out: str) -> None:
    lines = [
        f"# {cfg.name}", "",
        "A frozen-candidate subword tokenizer for the **BERT-Cord** research project "
        "(Tokenizer Milestone).", "",
        "> Scope: this is a tokenizer artifact only. It makes **no** model, coordination, or "
        "language-understanding claim, and is not a Transformers `AutoModel`. It is loadable "
        "with Hugging Face `tokenizers` / `PreTrainedTokenizerFast`.", "",
        "## Details", "",
        f"- Algorithm: **{cfg.algorithm}**",
        f"- Vocab size: requested {cfg.vocab_size:,}, actual "
        f"**{manifest['actual_vocab_size']:,}**",
        f"- Normalization: `{cfg.normalization}` · lowercase: {cfg.lowercase} · "
        f"byte_fallback: {cfg.byte_fallback}",
        f"- Special tokens (fixed ids): "
        + ", ".join(f"`{t}`={i}" for t, i in manifest['special_token_ids'].items()),
        f"- Git commit: `{manifest['git_commit']}`",
        f"- tokenizer.json SHA-256: `{manifest['tokenizer_json_sha256'][:16]}…`", "",
        "## Files", "",
        "- `tokenizer.json` — the tokenizer (HF `tokenizers` format)",
        "- `tokenizer_config.json` — `PreTrainedTokenizerFast` config",
        "- `special_tokens_map.json` — special-token mapping",
        "- `tokenizer_manifest.json` — provenance + checksums",
        "- `README.md` — this file", "",
        "## Usage", "",
        "```python",
        "from tokenizers import Tokenizer",
        f'tok = Tokenizer.from_file("{cfg.name}/tokenizer.json")',
        'enc = tok.encode("BERT-Cord tokenizer example.")',
        "print(enc.ids, enc.tokens)",
        "```", "",
        "_This tokenizer will be frozen and reused for all future MLM pretraining once "
        "selected. Trained for pipeline robustness, not final quality._",
    ]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
