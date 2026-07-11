"""Input embeddings: learned token + position + (optional) token-type embeddings."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .configuration import ModelConfig


class BertEmbeddings(nn.Module):
    """Sum of token, position, and (configurable) token-type embeddings + LayerNorm.

    Positions are absolute and learned, matching BERT-original. Token-type embeddings are
    only created when ``type_vocab_size > 0`` so single-segment models stay lean.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings: Optional[nn.Embedding]
        if config.use_token_type:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
        else:
            self.token_type_embeddings = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Non-persistent buffer of position ids [0, 1, ..., max_position-1].
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        embeddings = self.word_embeddings(input_ids)
        embeddings = embeddings + self.position_embeddings(position_ids)

        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
