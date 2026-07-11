"""Dynamic masked-language-model corruption.

Implements the standard BERT recipe applied *dynamically* (fresh masking each time a batch is
drawn, not baked into the dataset):

  * select ~15% of maskable (non-special) tokens,
  * of the selected tokens: 80% -> [MASK], 10% -> random token, 10% -> unchanged,
  * special tokens are never selected,
  * labels are -100 everywhere except at selected positions (so loss ignores the rest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class MaskingOutput:
    input_ids: torch.Tensor  # corrupted inputs
    labels: torch.Tensor     # -100 except at selected positions


class MLMasker:
    """Applies dynamic MLM masking to a batch of token ids."""

    def __init__(
        self,
        mask_token_id: int,
        vocab_size: int,
        special_token_ids: Sequence[int],
        mlm_probability: float = 0.15,
        pad_token_id: int = 0,
        replace_mask_prob: float = 0.8,
        replace_random_prob: float = 0.1,
    ) -> None:
        if not 0.0 < mlm_probability < 1.0:
            raise ValueError("mlm_probability must be in (0, 1).")
        if replace_mask_prob + replace_random_prob > 1.0:
            raise ValueError("replace_mask_prob + replace_random_prob must be <= 1.")
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_token_ids = list(special_token_ids)
        self.mlm_probability = mlm_probability
        self.pad_token_id = pad_token_id
        self.replace_mask_prob = replace_mask_prob
        self.replace_random_prob = replace_random_prob

    def special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Boolean mask that is True at positions holding a special token."""
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in self.special_token_ids:
            mask |= input_ids == tid
        return mask

    def __call__(
        self,
        input_ids: torch.Tensor,
        special_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> MaskingOutput:
        device = input_ids.device
        labels = input_ids.clone()

        if special_mask is None:
            special_mask = self.special_tokens_mask(input_ids)

        # Probability matrix for selection; special tokens get probability 0.
        prob = torch.full(input_ids.shape, self.mlm_probability, device=device)
        prob.masked_fill_(special_mask, 0.0)
        selected = torch.bernoulli(prob, generator=generator).bool()

        # Non-selected tokens are ignored by the loss.
        labels[~selected] = -100

        # 80% -> [MASK]
        mask_prob = torch.full(input_ids.shape, self.replace_mask_prob, device=device)
        replace_mask = torch.bernoulli(mask_prob, generator=generator).bool() & selected
        input_ids = input_ids.clone()
        input_ids[replace_mask] = self.mask_token_id

        # 10% -> random token (from the remaining selected-but-not-[MASK] tokens)
        rand_share = self.replace_random_prob / (1.0 - self.replace_mask_prob)
        rand_prob = torch.full(input_ids.shape, rand_share, device=device)
        replace_random = (
            torch.bernoulli(rand_prob, generator=generator).bool()
            & selected
            & ~replace_mask
        )
        random_tokens = torch.randint(
            self.vocab_size, input_ids.shape, device=device, generator=generator,
            dtype=input_ids.dtype,
        )
        input_ids[replace_random] = random_tokens[replace_random]

        # Remaining ~10% of selected tokens keep their original id (already unchanged).
        return MaskingOutput(input_ids=input_ids, labels=labels)
