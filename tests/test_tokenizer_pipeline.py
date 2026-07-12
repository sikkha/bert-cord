"""Tests for the tokenizer pipeline: corpus prep, training, loading, round-trip, specials."""

from __future__ import annotations

import json
import os

import pytest

from coordinator_bert.corpus import (
    CorpusConfig,
    document_language,
    prepare_corpus,
)
from coordinator_bert.tokenizer_eval import EXPECTED_SPECIAL_IDS, evaluate_tokenizer
from coordinator_bert.tokenizer_train import (
    SPECIAL_IDS,
    TokenizerTrainConfig,
    train_tokenizer,
)

pytest.importorskip("tokenizers")

_EN = [
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization converts raw text into subword units.",
    "Reproducible pipelines are robust, tested, and documented.",
    "Byte level BPE, WordPiece, and Unigram are subword algorithms.",
    "Special tokens include padding, classification, and mask.",
]
_TH = [
    "การแบ่งคำภาษาไทยเป็นงานที่ท้าทาย",
    "ตัวตัดคำแปลงข้อความดิบเป็นหน่วยย่อย",
]


def _write_corpus(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    # Repeat lines to exercise dedup; add a jsonl source.
    (d / "en.txt").write_text("\n".join(_EN + _EN[:2]) + "\n", encoding="utf-8")
    (d / "th.txt").write_text("\n".join(_TH) + "\n", encoding="utf-8")
    (d / "recs.jsonl").write_text(
        "\n".join(json.dumps({"text": t}) for t in
                  ["def f(x): return x", "Round-trip decoding matters."]) + "\n",
        encoding="utf-8")
    return str(d)


# --------------------------------------------------------------------------------------- #
# Corpus preparation
# --------------------------------------------------------------------------------------- #
def test_document_language():
    assert document_language("The quick brown fox") == "latin"
    assert document_language("การแบ่งคำภาษาไทย") == "thai"
    assert document_language("12345 %%%") == "unknown"


def test_prepare_corpus_dedup_shard_manifest(tmp_path):
    raw = _write_corpus(tmp_path)
    out = str(tmp_path / "corpus")
    cfg = CorpusConfig(shuffle_seed=7, val_fraction=0.2, shard_size=100, dedup=True)
    man = prepare_corpus([raw], out, cfg)
    st = man["statistics"]
    assert st["duplicates_removed"] == 2                 # the two repeated EN lines
    assert st["documents_kept"] == len(_EN) + len(_TH) + 2
    assert "thai" in st["per_language_docs"] and "latin" in st["per_language_docs"]
    # Files written.
    assert os.path.exists(os.path.join(out, "corpus_manifest.json"))
    assert os.path.exists(os.path.join(out, "corpus_report.md"))
    assert man["train_shards"] and man["train_shards"][0]["documents"] >= 1
    assert man["validation_documents"] >= 1


def test_prepare_corpus_deterministic(tmp_path):
    raw = _write_corpus(tmp_path)
    cfg = CorpusConfig(shuffle_seed=123, val_fraction=0.0, shard_size=100)
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    prepare_corpus([raw], a, cfg)
    prepare_corpus([raw], b, cfg)
    ta = open(os.path.join(a, "train-00000.txt"), encoding="utf-8").read()
    tb = open(os.path.join(b, "train-00000.txt"), encoding="utf-8").read()
    assert ta == tb  # same seed -> identical shuffle/order


# --------------------------------------------------------------------------------------- #
# Tokenizer training + loading + round-trip + specials
# --------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def corpus_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    lines = (_EN * 6) + (_TH * 6) + ["def f(x): return x", "# Heading\n"] * 3
    p = str(d / "train.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(s.replace("\n", " ") for s in lines) + "\n")
    return p


@pytest.mark.parametrize("algo", ["wordpiece", "byte_bpe", "unigram"])
def test_train_load_roundtrip_specials(algo, corpus_file, tmp_path):
    from tokenizers import Tokenizer

    cfg = TokenizerTrainConfig(name=f"t-{algo}", algorithm=algo, vocab_size=300,
                               min_frequency=1)
    out = str(tmp_path / algo)
    man = train_tokenizer(cfg, [corpus_file], out)

    # Artifact files.
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer_manifest.json", "README.md"):
        assert os.path.exists(os.path.join(out, f)), f
    # Manifest sanity.
    assert man["algorithm"] == algo and man["actual_vocab_size"] >= 5
    assert man["special_token_ids"] == SPECIAL_IDS

    # Loadable + special tokens at fixed ids.
    tok = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    for t, i in SPECIAL_IDS.items():
        assert tok.token_to_id(t) == i

    # Round-trip: whitespace-normalized decode reconstructs an ASCII sentence.
    text = "Tokenization converts raw text into subword units."
    enc = tok.encode(text)
    assert len(enc.ids) > 0
    dec = tok.decode(enc.ids, skip_special_tokens=True)
    import re
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    assert norm(dec) == norm(text)

    # Post-processor adds [CLS] .. [SEP].
    assert enc.ids[0] == SPECIAL_IDS["[CLS]"] and enc.ids[-1] == SPECIAL_IDS["[SEP]"]


def test_config_from_yaml(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(
        "tokenizer:\n  name: x\n  algorithm: unigram\n  vocab_size: 100\n"
        "  special_tokens: ['[PAD]','[CLS]','[SEP]','[MASK]','[UNK]']\n", encoding="utf-8")
    cfg = TokenizerTrainConfig.from_yaml(str(y))
    assert cfg.algorithm == "unigram" and cfg.vocab_size == 100


def test_config_rejects_bad_special_tokens():
    with pytest.raises(ValueError):
        TokenizerTrainConfig.from_dict({"special_tokens": ["[X]", "[CLS]"]})
    with pytest.raises(ValueError):
        TokenizerTrainConfig.from_dict({"algorithm": "sentencepiece"})


# --------------------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------------------- #
def test_evaluate_tokenizer_metrics(corpus_file, tmp_path):
    cfg = TokenizerTrainConfig(name="t-eval", algorithm="byte_bpe", vocab_size=300,
                               min_frequency=1)
    out = str(tmp_path / "tok")
    train_tokenizer(cfg, [corpus_file], out)
    metrics = evaluate_tokenizer(out, [corpus_file])
    for key in ("unknown_token_rate", "avg_tokens_per_sentence", "avg_tokens_per_word",
                "roundtrip_normalized_rate", "vocabulary_utilization",
                "reserved_token_integrity", "reserved_token_ids"):
        assert key in metrics
    assert metrics["reserved_token_integrity"] is True
    assert metrics["reserved_token_ids"] == EXPECTED_SPECIAL_IDS
    assert 0.0 <= metrics["unknown_token_rate"] <= 1.0
    # byte-level BPE reconstructs text -> high normalized round-trip.
    assert metrics["roundtrip_normalized_rate"] >= 0.9
