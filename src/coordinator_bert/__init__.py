"""coordinator_bert — a compact, custom BERT-style MLM system for coordination research.

Milestone 0 exposes only the base MLM stack. Distillation and coordination heads are
placeholders (see project_brief.md).
"""

from __future__ import annotations

from .configuration import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrainConfig,
    load_config,
)
from .masking import MLMasker, MaskingOutput
from .model import (
    BertForMaskedLM,
    BertModel,
    count_parameters,
    parameter_count_report,
)

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "OutputConfig",
    "RunConfig",
    "load_config",
    "BertModel",
    "BertForMaskedLM",
    "count_parameters",
    "parameter_count_report",
    "MLMasker",
    "MaskingOutput",
    "__version__",
]
