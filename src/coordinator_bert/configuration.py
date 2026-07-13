"""Configuration structures for the custom coordinator BERT.

Config is expressed as frozen dataclasses and loaded from YAML. Keeping the config as plain
dataclasses (rather than a framework config object) makes the architecture easy to inspect
and validate. A closed-form parameter-count estimator lives here so the target size can be
checked without instantiating the model.
"""

from __future__ import annotations

import os
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
    # Directory of a pre-tokenized, packed corpus (data/tokenized/<run>/). When set, the packed
    # loader is used (highest-priority dispatch); otherwise dataset_name / synthetic apply.
    packed_dataset_dir: Optional[str] = None
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
class RuntimeConfig:
    """Hardware / performance knobs — deliberately separate from the scientific settings.

    These never change the mathematical workload; they select the device and toggle optional
    performance features that are feature-detected and safely disabled when unavailable (see
    ``coordinator_bert.runtime``).
    """

    device: str = "auto"                # auto | cpu | cuda | mps
    allow_tf32: bool = False            # TF32 matmul/cudnn on Ampere+ CUDA
    pin_memory: bool = False            # CUDA host->device pinned staging
    num_workers: int = 0                # dataloader workers
    persistent_workers: bool = False    # only honored when num_workers > 0
    non_blocking: bool = False          # async H2D copies (CUDA only)
    fused_adamw: bool = False           # fused AdamW when available (CUDA)
    torch_compile: bool = False         # opt-in only; never default
    compile_mode: str = "default"       # torch.compile mode when enabled

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass(frozen=True)
class TrackingConfig:
    """Optional experiment tracking. Default backend is ``none`` (a no-op).

    Never affects training mathematics or the local JSONL/report pipeline. No secrets belong
    here — online mode uses W&B's normal login / WANDB_API_KEY mechanism.
    """

    backend: str = "none"           # none | wandb
    mode: str = "offline"           # offline | online | disabled
    project: str = "bert-cord"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "training"
    tags: tuple = ()
    notes: Optional[str] = None
    log_interval: int = 10
    log_code: bool = False
    log_checkpoints: bool = False
    log_analysis_artifacts: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrackingConfig":
        d = dict(d or {})
        if isinstance(d.get("tags"), list):
            d["tags"] = tuple(d["tags"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class RunConfig:
    """Top-level container binding model + train + data + output + runtime configs."""

    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    output: OutputConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            train=TrainConfig.from_dict(d.get("train", {})),
            data=DataConfig.from_dict(d.get("data", {})),
            output=OutputConfig.from_dict(d.get("output", {})),
            runtime=RuntimeConfig.from_dict(d.get("runtime", {})),
            tracking=TrackingConfig.from_dict(d.get("tracking", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        return cls.from_dict(load_config_dict(path))

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins; dicts merge, scalars replace)."""
    out = dict(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_dict(path: str, _seen: Optional[set] = None) -> dict:
    """Load a YAML config dict, resolving an optional ``extends:`` composition.

    ``extends`` may be a string or list of paths (relative to the current file). Bases are
    deep-merged in order, then the current file's own keys overlay them. This lets a resolved
    config compose a model file + a platform file without the loader needing multiple args,
    while plain flat configs (no ``extends``) load exactly as before.
    """
    path = os.path.abspath(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular config extends detected at {path}")
    _seen = _seen | {path}

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping at top level")

    extends = raw.pop("extends", None)
    if not extends:
        return raw

    if isinstance(extends, str):
        extends = [extends]
    base_dir = os.path.dirname(path)
    merged: dict = {}
    for rel in extends:
        base_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        merged = _deep_merge(merged, load_config_dict(base_path, _seen))
    return _deep_merge(merged, raw)


def load_config(path: str) -> RunConfig:
    """Convenience loader used by scripts (supports flat configs and ``extends`` composition)."""
    return RunConfig.from_yaml(path)
