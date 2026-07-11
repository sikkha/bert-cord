"""Pytest fixtures and path setup for the coordinator_bert test suite."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coordinator_bert.configuration import ModelConfig  # noqa: E402


@pytest.fixture
def tiny_config() -> ModelConfig:
    """A small model config that is fast to instantiate and run on CPU."""
    return ModelConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_type="post",
        tie_word_embeddings=True,
    )


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield
