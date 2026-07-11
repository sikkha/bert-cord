"""Masked-language-model prediction head.

Transform (dense + activation + LayerNorm) followed by a decoder that projects hidden states
back to vocabulary logits. The decoder weight is tied to the input token-embedding matrix when
``tie_word_embeddings`` is set; an independent output bias is always learned.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import ModelConfig

_ACT = {"gelu": F.gelu, "relu": F.relu}


class BertMLMHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.act = _ACT[config.hidden_act]
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # decoder.weight is set/tied by the parent model; bias is always independent.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return self.decoder(hidden_states)
