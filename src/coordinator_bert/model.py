"""Custom BERT encoder, pooler, and masked-language-model wrapper.

Implemented from scratch (no Hugging Face ``BertModel`` / ``AutoModelForMaskedLM``). Supports
both post-layer-norm (BERT-original) and pre-layer-norm blocks, selected via config.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .attention import MultiHeadSelfAttention, build_extended_attention_mask
from .configuration import ModelConfig
from .embeddings import BertEmbeddings
from .mlm_head import BertMLMHead

_ACT = {"gelu": F.gelu, "relu": F.relu}


class FeedForward(nn.Module):
    """Position-wise feed-forward network (dense -> activation -> dense)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.act = _ACT[config.hidden_act]
        self.fc_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc_in(x))
        x = self.fc_out(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """One encoder block with residual connections around attention and FFN.

    post-LN:  x = LN(x + Sublayer(x))
    pre-LN:   x = x + Sublayer(LN(x))
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_type = config.norm_type
        self.attention = MultiHeadSelfAttention(config)
        self.ffn = FeedForward(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        extended_attention_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.norm_type == "post":
            attn_out, probs = self.attention(
                hidden_states, extended_attention_mask, return_probs=return_probs
            )
            hidden_states = self.attn_norm(hidden_states + attn_out)
            ffn_out = self.ffn(hidden_states)
            hidden_states = self.ffn_norm(hidden_states + ffn_out)
        else:  # pre-LN
            attn_out, probs = self.attention(
                self.attn_norm(hidden_states), extended_attention_mask,
                return_probs=return_probs,
            )
            hidden_states = hidden_states + attn_out
            ffn_out = self.ffn(self.ffn_norm(hidden_states))
            hidden_states = hidden_states + ffn_out
        return hidden_states, probs


class BertEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # A final LayerNorm is required for pre-LN so outputs are normalized.
        self.final_norm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.norm_type == "pre"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        extended_attention_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_probs: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, probs = layer(
                hidden_states, extended_attention_mask, return_probs=return_probs
            )
            if return_probs and probs is not None:
                all_probs.append(probs)
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        return hidden_states, all_probs


class BertPooler(nn.Module):
    """Pools the [CLS] (position 0) hidden state through a tanh dense layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dense(hidden_states[:, 0]))


class BertModel(nn.Module):
    """Custom bidirectional encoder returning sequence output and pooled output."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> dict[str, object]:
        embeddings = self.embeddings(input_ids, token_type_ids, position_ids)
        ext_mask = build_extended_attention_mask(attention_mask, embeddings.dtype)
        sequence_output, all_probs = self.encoder(
            embeddings, ext_mask, return_probs=return_probs
        )
        pooled_output = self.pooler(sequence_output)
        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "attention_probs": all_probs,
        }


class BertForMaskedLM(nn.Module):
    """Encoder + MLM head with tied input/output token embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.mlm_head = BertMLMHead(config)
        self.mlm_head.apply(self.bert._init_weights)
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        """Share the decoder weight matrix with the input token-embedding matrix."""
        self.mlm_head.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.bert.embeddings.word_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.mlm_head.decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> dict[str, object]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_probs=return_probs,
        )
        logits = self.mlm_head(outputs["last_hidden_state"])

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": outputs["last_hidden_state"],
            "attention_probs": outputs["attention_probs"],
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and (de-duplicated) unique parameters.

    Tied parameters share storage; ``unique`` counts each storage once so tied embeddings are
    not double-counted.
    """
    total = 0
    trainable = 0
    seen: set[int] = set()
    unique = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if id(p) not in seen:
            seen.add(id(p))
            unique += n
    return {"total": total, "trainable": trainable, "unique": unique}


def parameter_count_report(model: nn.Module) -> str:
    c = count_parameters(model)
    return (
        f"parameters: total={c['total']:,} trainable={c['trainable']:,} "
        f"unique(dedup tied)={c['unique']:,} (~{c['unique'] / 1e6:.2f}M)"
    )
