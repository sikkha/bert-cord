"""Configuration structures for the custom coordinator BERT.

Config is expressed as frozen dataclasses and loaded from YAML. Keeping the config as plain
dataclasses (rather than a framework config object) makes the architecture easy to inspect
and validate. A closed-form parameter-count estimator lives here so the target size can be
checked without instantiating the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters for the encoder."""

    vocab_size: int = 32000
    hidden_size: int = 384
    num_hidden_layers: int = 8
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 512
    type_vocab_size: int = 2  # 0 disables token-type embeddings
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    norm_type: str = "post"  # "post" or "pre"
    tie_word_embeddings: bool = True
    use_sdpa: bool = False
    pad_token_id: int = 0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.norm_type not in ("post", "pre"):
            raise ValueError(f"norm_type must be 'post' or 'pre', got {self.norm_type!r}.")
        if self.hidden_act not in ("gelu", "relu"):
            raise ValueError(f"hidden_act must be 'gelu' or 'relu', got {self.hidden_act!r}.")
        if self.type_vocab_size < 0:
            raise ValueError("type_vocab_size must be >= 0.")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def use_token_type(self) -> bool:
        return self.type_vocab_size > 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def estimate_num_parameters(self) -> int:
        """Closed-form parameter estimate (tied embeddings, learnable LayerNorm)."""
        h = self.hidden_size
        # Embeddings
        emb = self.vocab_size * h + self.max_position_embeddings * h
        if self.use_token_type:
            emb += self.type_vocab_size * h
        emb += 2 * h  # embedding LayerNorm (weight + bias)

        # One transformer layer
        attn = 4 * (h * h + h)  # Q, K, V, O (weight + bias)
        attn_ln = 2 * h
        ffn = (h * self.intermediate_size + self.intermediate_size) + (
            self.intermediate_size * h + h
        )
        ffn_ln = 2 * h
        per_layer = attn + attn_ln + ffn + ffn_ln
        layers = self.num_hidden_layers * per_layer

        # Pooler
        pooler = h * h + h

        # MLM head: transform dense + LN + decoder bias (decoder weight tied to embeddings)
        mlm_head = (h * h + h) + 2 * h + self.vocab_size
        if not self.tie_word_embeddings:
            mlm_head += self.vocab_size * h

        # Final pre-LN encoder LayerNorm (only present for pre-norm)
        final_ln = 2 * h if self.norm_type == "pre" else 0

        return emb + layers + pooler + mlm_head + final_ln


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    precision: str = "auto"  # auto | bf16 | fp16 | fp32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 2000
    lr_scheduler: str = "cosine"  # cosine | linear
    min_lr_ratio: float = 0.1
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 128
    mlm_probability: float = 0.15
    eval_every: int = 100
    save_every: int = 200
    eval_max_batches: int = 20
    log_every: int = 20

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SyntheticConfig:
    num_train_examples: int = 4096
    num_val_examples: int = 512
    min_len: int = 24
    max_len: int = 128


@dataclass(frozen=True)
class DataConfig:
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    text_column: str = "text"
    tokenizer_path: Optional[str] = None
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataConfig":
        d = dict(d)
        syn = d.pop("synthetic", {}) or {}
        known = {f.name for f in fields(cls)} - {"synthetic"}
        return cls(
            synthetic=SyntheticConfig(**{k: v for k, v in syn.items()
                                         if k in {f.name for f in fields(SyntheticConfig)}}),
            **{k: v for k, v in d.items() if k in known},
        )


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "experiments/run"
    checkpoint_dir: str = "experiments/run/checkpoints"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OutputConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class RunConfig:
    """Top-level container binding model + train + data + output configs."""

    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            train=TrainConfig.from_dict(d.get("train", {})),
            data=DataConfig.from_dict(d.get("data", {})),
            output=OutputConfig.from_dict(d.get("output", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)


def load_config(path: str) -> RunConfig:
    """Convenience loader used by scripts."""
    return RunConfig.from_yaml(path)
