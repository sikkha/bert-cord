"""Dataset preparation and dataloaders for MLM pretraining.

Two paths are supported:

  * **synthetic** (default, offline): deterministic random token sequences with proper
    [CLS]/[SEP] framing. Used for smoke tests and CI — no network, no tokenizer files.
  * **text dataset** (optional): a Hugging Face ``datasets`` corpus tokenized with a Hugging
    Face tokenizer. Enabled by setting ``data.dataset_name`` and ``data.tokenizer_path``.

Masking is applied *dynamically* in the collator (see ``masking.py``), so each epoch sees
fresh corruption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .configuration import DataConfig, ModelConfig, TrainConfig
from .masking import MLMasker


@dataclass(frozen=True)
class SpecialTokens:
    """Reserved special-token ids. Real vocabulary starts at ``first_real_id``."""

    pad: int = 0
    cls: int = 1
    sep: int = 2
    mask: int = 3
    unk: int = 4

    @property
    def all_ids(self) -> list[int]:
        return [self.pad, self.cls, self.sep, self.mask, self.unk]

    @property
    def first_real_id(self) -> int:
        return max(self.all_ids) + 1


class TokenIdDataset(Dataset):
    """Holds pre-tokenized examples as lists of token ids (already framed with CLS/SEP)."""

    def __init__(self, examples: Sequence[Sequence[int]]) -> None:
        self.examples = [list(map(int, ex)) for ex in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> list[int]:
        return self.examples[idx]


def build_synthetic_examples(
    num_examples: int,
    vocab_size: int,
    specials: SpecialTokens,
    min_len: int,
    max_len: int,
    seed: int,
    period: int = 3,
    motif_vocab: int = 64,
) -> list[list[int]]:
    """Generate deterministic, *learnable* synthetic sequences.

    Each sequence body is a short random motif of length ``period`` tiled to fill the
    sequence, then framed as ``[CLS] body... [SEP]``. Because a masked token can be recovered
    by copying the token ``period`` positions away (usually unmasked), an MLM has real signal:
    training loss should fall and masked-token accuracy should rise. This makes the smoke run
    a genuine end-to-end correctness check of the gradient/optimizer path — not just a
    no-op over random noise.

    ``motif_vocab`` caps the token range used for motifs (kept well below ``vocab_size``) so a
    tiny model can learn the mapping quickly during a short smoke run.
    """
    g = torch.Generator().manual_seed(seed)
    lo = specials.first_real_id
    hi = min(vocab_size, lo + max(2, motif_vocab))
    period = max(1, min(period, max_len - 2))
    examples: list[list[int]] = []
    for _ in range(num_examples):
        length = int(torch.randint(min_len, max_len + 1, (1,), generator=g).item())
        body_len = max(1, length - 2)
        motif = torch.randint(lo, hi, (period,), generator=g).tolist()
        body = [motif[i % period] for i in range(body_len)]
        examples.append([specials.cls, *body, specials.sep])
    return examples


class MLMCollator:
    """Pads a batch, builds attention masks, and applies dynamic MLM masking."""

    def __init__(
        self,
        masker: MLMasker,
        pad_token_id: int,
        max_seq_length: int,
        seed: Optional[int] = None,
    ) -> None:
        self.masker = masker
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)

    def __call__(self, batch: list[list[int]]) -> dict[str, torch.Tensor]:
        truncated = [ex[: self.max_seq_length] for ex in batch]
        max_len = max(len(ex) for ex in truncated)
        input_ids = torch.full((len(truncated), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(truncated), max_len), dtype=torch.long)
        for i, ex in enumerate(truncated):
            input_ids[i, : len(ex)] = torch.tensor(ex, dtype=torch.long)
            attention_mask[i, : len(ex)] = 1

        # Treat padding as a special token so it is never selected for masking.
        special_mask = self.masker.special_tokens_mask(input_ids)
        special_mask |= attention_mask == 0
        masked = self.masker(input_ids, special_mask=special_mask, generator=self._generator)
        return {
            "input_ids": masked.input_ids,
            "attention_mask": attention_mask,
            "labels": masked.labels,
        }


def build_masker(model_cfg: ModelConfig, train_cfg: TrainConfig,
                 specials: SpecialTokens) -> MLMasker:
    return MLMasker(
        mask_token_id=specials.mask,
        vocab_size=model_cfg.vocab_size,
        special_token_ids=specials.all_ids,
        mlm_probability=train_cfg.mlm_probability,
        pad_token_id=specials.pad,
    )


def _loader_kwargs(runtime=None) -> dict:
    """DataLoader performance kwargs from a resolved runtime (empty when none given).

    ``runtime`` is a ``coordinator_bert.runtime.ResolvedRuntime`` whose values are already
    feature-checked (persistent_workers only when num_workers>0, pin_memory only on CUDA).
    """
    if runtime is None:
        return {}
    kwargs = {"num_workers": int(getattr(runtime, "num_workers", 0)),
              "pin_memory": bool(getattr(runtime, "pin_memory", False))}
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(getattr(runtime, "persistent_workers", False))
    return kwargs


def build_synthetic_dataloaders(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    specials: Optional[SpecialTokens] = None,
    runtime=None,
) -> tuple[DataLoader, DataLoader, SpecialTokens]:
    """Build train/val dataloaders backed by the synthetic corpus."""
    specials = specials or SpecialTokens()
    syn = data_cfg.synthetic
    train_examples = build_synthetic_examples(
        syn.num_train_examples, model_cfg.vocab_size, specials,
        syn.min_len, min(syn.max_len, train_cfg.max_seq_length), seed=train_cfg.seed,
    )
    val_examples = build_synthetic_examples(
        syn.num_val_examples, model_cfg.vocab_size, specials,
        syn.min_len, min(syn.max_len, train_cfg.max_seq_length), seed=train_cfg.seed + 1,
    )
    masker = build_masker(model_cfg, train_cfg, specials)
    train_collator = MLMCollator(masker, specials.pad, train_cfg.max_seq_length,
                                 seed=train_cfg.seed)
    # Fixed generator seed for eval so validation masking is reproducible.
    val_collator = MLMCollator(masker, specials.pad, train_cfg.max_seq_length,
                               seed=train_cfg.seed + 12345)

    lk = _loader_kwargs(runtime)
    train_loader = DataLoader(
        TokenIdDataset(train_examples),
        batch_size=train_cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=train_collator,
        drop_last=True,
        **lk,
    )
    val_loader = DataLoader(
        TokenIdDataset(val_examples),
        batch_size=train_cfg.per_device_batch_size,
        shuffle=False,
        collate_fn=val_collator,
        drop_last=False,
        **lk,
    )
    return train_loader, val_loader, specials


def build_text_dataloaders(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    runtime=None,
) -> tuple[DataLoader, DataLoader, SpecialTokens]:
    """Build dataloaders from a Hugging Face text dataset + tokenizer (optional path).

    Requires ``datasets`` and ``tokenizers``. Kept intentionally small and explicit; only
    used when ``data.dataset_name`` and ``data.tokenizer_path`` are provided.
    """
    from datasets import load_dataset  # local import: optional dependency
    from tokenizers import Tokenizer

    if not data_cfg.tokenizer_path:
        raise ValueError("data.tokenizer_path is required for the text-dataset path.")
    tok = Tokenizer.from_file(data_cfg.tokenizer_path)

    def special_id(name: str, default: int) -> int:
        tid = tok.token_to_id(name)
        return default if tid is None else tid

    specials = SpecialTokens(
        pad=special_id("[PAD]", 0),
        cls=special_id("[CLS]", 1),
        sep=special_id("[SEP]", 2),
        mask=special_id("[MASK]", 3),
        unk=special_id("[UNK]", 4),
    )

    ds = load_dataset(data_cfg.dataset_name, data_cfg.dataset_config)
    col = data_cfg.text_column

    def encode(example: dict) -> dict:
        enc = tok.encode(example[col])
        return {"ids": enc.ids[: train_cfg.max_seq_length]}

    train_split = ds["train"].map(encode, remove_columns=ds["train"].column_names)
    val_key = "validation" if "validation" in ds else "test"
    val_split = ds[val_key].map(encode, remove_columns=ds[val_key].column_names)

    train_examples = [ex["ids"] for ex in train_split if len(ex["ids"]) > 2]
    val_examples = [ex["ids"] for ex in val_split if len(ex["ids"]) > 2]

    masker = build_masker(model_cfg, train_cfg, specials)
    train_collator = MLMCollator(masker, specials.pad, train_cfg.max_seq_length,
                                 seed=train_cfg.seed)
    val_collator = MLMCollator(masker, specials.pad, train_cfg.max_seq_length,
                               seed=train_cfg.seed + 12345)
    lk = _loader_kwargs(runtime)
    train_loader = DataLoader(
        TokenIdDataset(train_examples), batch_size=train_cfg.per_device_batch_size,
        shuffle=True, collate_fn=train_collator, drop_last=True, **lk,
    )
    val_loader = DataLoader(
        TokenIdDataset(val_examples), batch_size=train_cfg.per_device_batch_size,
        shuffle=False, collate_fn=val_collator, drop_last=False, **lk,
    )
    return train_loader, val_loader, specials


class PackedMLMCollator:
    """Dynamic MLM masking for already-packed fixed-length rows.

    Reuses the same :class:`MLMasker` as the other paths (identical MLM objective). Unlike
    :class:`MLMCollator`, the attention mask is derived from the pad id (rows are pre-padded to
    a fixed length with ``[CLS] content [SEP] PAD...``), and special tokens (including PAD) are
    never selected for masking.
    """

    def __init__(self, masker: MLMasker, pad_token_id: int, seed: Optional[int] = None) -> None:
        self.masker = masker
        self.pad_token_id = pad_token_id
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        rows = [b if isinstance(b, torch.Tensor) else torch.as_tensor(b, dtype=torch.long)
                for b in batch]
        input_ids = torch.stack(rows).to(torch.long)
        attention_mask = (input_ids != self.pad_token_id).to(torch.long)
        special_mask = self.masker.special_tokens_mask(input_ids)
        special_mask |= attention_mask == 0
        masked = self.masker(input_ids, special_mask=special_mask, generator=self._generator)
        return {
            "input_ids": masked.input_ids,
            "attention_mask": attention_mask,
            "labels": masked.labels,
        }


def build_packed_dataloaders(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    specials: Optional[SpecialTokens] = None,
    runtime=None,
) -> tuple[DataLoader, DataLoader, SpecialTokens]:
    """Build dataloaders over a pre-tokenized, memory-mapped packed corpus.

    Uses the existing :class:`MLMasker` for dynamic masking (unchanged MLM objective) and the
    same runtime DataLoader options. Validation masking is deterministic (fixed generator seed).
    """
    from .packed_corpus import PackedTokenDataset

    specials = specials or SpecialTokens()
    packed_dir = data_cfg.packed_dataset_dir
    train_ds = PackedTokenDataset(packed_dir, "train")
    if len(train_ds) == 0:
        raise ValueError(f"packed train split is empty: {packed_dir}")
    seq_len = train_ds.sequence_length
    if seq_len > model_cfg.max_position_embeddings:
        raise ValueError(f"packed sequence_length {seq_len} exceeds model "
                         f"max_position_embeddings {model_cfg.max_position_embeddings}")
    if seq_len != train_cfg.max_seq_length:
        print(f"[data] note: packed sequence_length {seq_len} != train.max_seq_length "
              f"{train_cfg.max_seq_length}; using the packed length {seq_len}.")

    val_ds = PackedTokenDataset(packed_dir, "validation")
    if len(val_ds) == 0:
        print("[data] note: packed corpus has no validation split; using the train split for "
              "validation (deterministic masking).")
        val_ds = train_ds

    masker = build_masker(model_cfg, train_cfg, specials)
    train_collator = PackedMLMCollator(masker, specials.pad, seed=train_cfg.seed)
    val_collator = PackedMLMCollator(masker, specials.pad, seed=train_cfg.seed + 12345)
    lk = _loader_kwargs(runtime)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.per_device_batch_size, shuffle=True,
        collate_fn=train_collator, drop_last=True, **lk,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.per_device_batch_size, shuffle=False,
        collate_fn=val_collator, drop_last=False, **lk,
    )
    return train_loader, val_loader, specials


def build_dataloaders(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    runtime=None,
) -> tuple[DataLoader, DataLoader, SpecialTokens]:
    """Dispatch to the packed / text-dataset / synthetic path based on config.

    Priority: ``packed_dataset_dir`` > ``dataset_name`` (HF text) > synthetic.
    """
    if data_cfg.packed_dataset_dir:
        return build_packed_dataloaders(model_cfg, train_cfg, data_cfg, runtime=runtime)
    if data_cfg.dataset_name:
        return build_text_dataloaders(model_cfg, train_cfg, data_cfg, runtime=runtime)
    return build_synthetic_dataloaders(model_cfg, train_cfg, data_cfg, runtime=runtime)
