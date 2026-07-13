"""Tests for the offline packed-corpus pipeline + memory-mapped dataset + dispatch."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

pytest.importorskip("tokenizers")

from coordinator_bert.configuration import DataConfig, ModelConfig, TrainConfig
from coordinator_bert.packed_corpus import (
    PackedTokenDataset,
    dtype_for_vocab,
    pack_corpus,
)
import validate_packed_corpus as vpc  # noqa: E402

PAD, CLS, SEP, MASK = 0, 1, 2, 3
VOCAB = 10  # [PAD,CLS,SEP,MASK,UNK] + a,b,c,d,e


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    """A controlled WordLevel tokenizer: a=5 b=6 c=7 d=8 e=9; specials 0-4."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4,
             "a": 5, "b": 6, "c": 7, "d": 8, "e": 9}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    d = tmp_path_factory.mktemp("tok")
    tok.save(os.path.join(str(d), "tokenizer.json"))
    return str(d)


def _content_between_cls_sep(row: list[int]) -> list[int]:
    assert row[0] == CLS
    sep = row.index(SEP)
    return row[1:sep]


# --------------------------------------------------------------------------------------- #
# dtype selection
# --------------------------------------------------------------------------------------- #
def test_dtype_selection():
    assert dtype_for_vocab(500) == "uint16"
    assert dtype_for_vocab(65535) == "uint16"
    assert dtype_for_vocab(65536) == "uint32"
    assert dtype_for_vocab(70000) == "uint32"


# --------------------------------------------------------------------------------------- #
# Packing correctness
# --------------------------------------------------------------------------------------- #
def test_token_preservation_and_framing(tokenizer_dir, tmp_path):
    train = tmp_path / "train.txt"
    train.write_text("a b c d e a b c d e\n", encoding="utf-8")  # 10 tokens -> multiple chunks
    out = str(tmp_path / "packed")
    pack_corpus(tokenizer_dir, [str(train)], [], out, sequence_length=5,
                sequences_per_shard=100, expected_vocab_size=VOCAB)
    arr = np.load(os.path.join(out, "train", "shard-00000.npy"))
    assert arr.dtype == np.uint16
    # Reconstruct all content across the document's chunks -> original 10 ids.
    content = []
    for row in arr.tolist():
        assert row[0] == CLS and SEP in row
        content += _content_between_cls_sep(row)
    assert content == [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]
    # Final chunk is padded; no MASK anywhere.
    assert (arr == MASK).sum() == 0
    assert arr[-1].tolist().count(PAD) > 0


def test_no_cross_document_packing(tokenizer_dir, tmp_path):
    train = tmp_path / "t.txt"
    # doc1 = 4 tokens (spills into 2 chunks: [a b c][d]); doc2 = 3 tokens ([e e e]).
    train.write_text("a b c d\ne e e\n", encoding="utf-8")
    out = str(tmp_path / "p")
    pack_corpus(tokenizer_dir, [str(train)], [], out, sequence_length=5,
                sequences_per_shard=100, expected_vocab_size=VOCAB)
    arr = np.load(os.path.join(out, "train", "shard-00000.npy")).tolist()
    assert len(arr) == 3
    # Row 1 = doc1 remainder [CLS, d, SEP, PAD, PAD] -> padded, NOT filled with doc2 tokens.
    assert _content_between_cls_sep(arr[1]) == [8]
    assert arr[1][3] == PAD and arr[1][4] == PAD
    # Row 2 = doc2 fully.
    assert _content_between_cls_sep(arr[2]) == [9, 9, 9]


def test_deterministic_packing(tokenizer_dir, tmp_path):
    train = tmp_path / "t.txt"
    train.write_text("a b c\nd e a\nb c d\n", encoding="utf-8")
    a = str(tmp_path / "a"); b = str(tmp_path / "b")
    ma = pack_corpus(tokenizer_dir, [str(train)], [], a, sequence_length=6,
                     sequences_per_shard=2, expected_vocab_size=VOCAB)
    mb = pack_corpus(tokenizer_dir, [str(train)], [], b, sequence_length=6,
                     sequences_per_shard=2, expected_vocab_size=VOCAB)
    # Same sequence counts and identical shard checksums.
    assert ma["train_sequences"] == mb["train_sequences"]
    ca = [s["sha256"] for s in ma["shards"]["train"]]
    cb = [s["sha256"] for s in mb["shards"]["train"]]
    assert ca == cb and len(ca) >= 2  # shard rollover happened


def test_vocab_mismatch_fails_loudly(tokenizer_dir, tmp_path):
    train = tmp_path / "t.txt"
    train.write_text("a b c\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pack_corpus(tokenizer_dir, [str(train)], [], str(tmp_path / "o"),
                    sequence_length=5, expected_vocab_size=32000)  # tokenizer vocab is 10


def test_overwrite_required(tokenizer_dir, tmp_path):
    train = tmp_path / "t.txt"
    train.write_text("a b\n", encoding="utf-8")
    out = str(tmp_path / "o")
    pack_corpus(tokenizer_dir, [str(train)], [], out, sequence_length=5,
                expected_vocab_size=VOCAB)
    with pytest.raises(FileExistsError):
        pack_corpus(tokenizer_dir, [str(train)], [], out, sequence_length=5,
                    expected_vocab_size=VOCAB)  # exists, no overwrite


# --------------------------------------------------------------------------------------- #
# Memory-mapped dataset
# --------------------------------------------------------------------------------------- #
def test_packed_dataset_indexing_across_shards(tokenizer_dir, tmp_path):
    train = tmp_path / "t.txt"
    train.write_text("\n".join(["a b c d e"] * 6) + "\n", encoding="utf-8")
    out = str(tmp_path / "p")
    pack_corpus(tokenizer_dir, [str(train)], [], out, sequence_length=4,
                sequences_per_shard=2, expected_vocab_size=VOCAB)  # -> several shards
    ds = PackedTokenDataset(out, "train")
    # Concatenate shards manually for the ground truth.
    import glob
    shards = sorted(glob.glob(os.path.join(out, "train", "*.npy")))
    full = np.concatenate([np.load(s) for s in shards], axis=0)
    assert len(ds) == full.shape[0] and len(shards) >= 2
    for i in (0, 1, 2, len(ds) - 1):
        row = ds[i]
        assert isinstance(row, torch.Tensor) and row.dtype == torch.long
        assert row.tolist() == full[i].tolist()


# --------------------------------------------------------------------------------------- #
# Validator (checksum tamper)
# --------------------------------------------------------------------------------------- #
def test_validator_passes_and_detects_corruption(tokenizer_dir, tmp_path):
    train = tmp_path / "t.txt"
    val = tmp_path / "v.txt"
    train.write_text("a b c d\ne a b c\n", encoding="utf-8")
    val.write_text("c d e\n", encoding="utf-8")
    out = str(tmp_path / "p")
    pack_corpus(tokenizer_dir, [str(train)], [str(val)], out, sequence_length=5,
                expected_vocab_size=VOCAB)
    assert vpc.validate(out, tokenizer_dir, require_validation=True).ok

    # Corrupt a shard -> checksum + framing checks should now fail.
    shard = os.path.join(out, "train", "shard-00000.npy")
    with open(shard, "ab") as fh:
        fh.write(b"\x00\x00")
    assert not vpc.validate(out, tokenizer_dir, require_validation=True).ok


# --------------------------------------------------------------------------------------- #
# Dispatch + dynamic masking + existing paths
# --------------------------------------------------------------------------------------- #
def _tiny_model():
    return ModelConfig(vocab_size=VOCAB, hidden_size=16, num_hidden_layers=1,
                       num_attention_heads=2, intermediate_size=32,
                       max_position_embeddings=32, type_vocab_size=2)


def test_dispatch_to_packed_and_dynamic_masking(tokenizer_dir, tmp_path):
    from coordinator_bert.data import build_dataloaders

    train = tmp_path / "t.txt"
    train.write_text("\n".join(["a b c d e a b c"] * 8) + "\n", encoding="utf-8")
    out = str(tmp_path / "p")
    pack_corpus(tokenizer_dir, [str(train)], [], out, sequence_length=8,
                sequences_per_shard=100, expected_vocab_size=VOCAB)

    mc = _tiny_model()
    tc = TrainConfig(max_seq_length=8, per_device_batch_size=4, seed=0, mlm_probability=0.5)
    dc = DataConfig(packed_dataset_dir=out)
    tl, vl, sp = build_dataloaders(mc, tc, dc)
    from coordinator_bert.packed_corpus import PackedTokenDataset as _PD
    assert isinstance(tl.dataset, _PD)
    batch = next(iter(tl))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == (4, 8)
    # Dynamic masking happened (supervised labels exist).
    assert int((batch["labels"] != -100).sum()) > 0
    # attention_mask is derived from the ORIGINAL (pre-mask) ids: [CLS] at col 0 is always
    # attended, and every supervised (labels != -100) position is attended.
    assert torch.all(batch["attention_mask"][:, 0] == 1)
    supervised = batch["labels"] != -100
    assert torch.all(batch["attention_mask"][supervised] == 1)
    # Labels only supervise real content positions (never PAD/CLS/SEP originals are irrelevant
    # here; just confirm labels are valid vocab ids where supervised).
    assert int((batch["labels"][supervised] >= 0).all())


def test_dispatch_priority_and_existing_paths(monkeypatch, tmp_path):
    import coordinator_bert.data as data

    mc = _tiny_model()
    tc = TrainConfig(max_seq_length=16, per_device_batch_size=4, seed=1)

    # packed_dataset_dir wins over dataset_name.
    sentinel_text = object()
    sentinel_packed = object()
    monkeypatch.setattr(data, "build_text_dataloaders",
                        lambda *a, **k: ("text", "text", None))
    monkeypatch.setattr(data, "build_packed_dataloaders",
                        lambda *a, **k: ("packed", "packed", None))
    r = data.build_dataloaders(mc, tc, DataConfig(packed_dataset_dir="x", dataset_name="y"))
    assert r[0] == "packed"
    # dataset_name -> text path (packed unset).
    r = data.build_dataloaders(mc, tc, DataConfig(dataset_name="y"))
    assert r[0] == "text"


def test_synthetic_path_unaffected():
    from coordinator_bert.data import build_dataloaders
    mc = _tiny_model()
    tc = TrainConfig(max_seq_length=16, per_device_batch_size=4, seed=2)
    from coordinator_bert.configuration import DataConfig as DC, SyntheticConfig
    dc = DC(synthetic=SyntheticConfig(num_train_examples=16, num_val_examples=8,
                                      min_len=8, max_len=16))
    tl, vl, sp = build_dataloaders(mc, tc, dc)
    batch = next(iter(tl))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
