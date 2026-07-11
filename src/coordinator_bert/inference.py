"""Inference helpers for Milestone 0.5 evaluation utilities.

These are thin, well-tested helpers used by the evaluation scripts (predict_mask,
overfit_tiny, evaluate_synthetic) and by tests. They do **not** change the model
architecture — they only load checkpoints and post-process logits.

Scope note: this milestone validates *inference mechanics*, *overfitting capacity*, and
*synthetic generalization*. Nothing here demonstrates language understanding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .checkpointing import load_checkpoint, resolve_checkpoint_path
from .configuration import ModelConfig
from .model import BertForMaskedLM


def load_model_for_inference(
    config: ModelConfig,
    checkpoint_path: Optional[str] = None,
    map_location: str = "cpu",
) -> BertForMaskedLM:
    """Build a model and (optionally) load weights, returning it in eval mode.

    Only model weights are restored (no optimizer/scheduler/RNG), which is what inference
    needs. ``eval()`` disables dropout so inference is deterministic.
    """
    model = BertForMaskedLM(config)
    if checkpoint_path is not None:
        # Accept a step dir, a state.pt, or a checkpoint root (resolved via latest.json).
        checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            map_location=map_location,
            restore_rng=False,
        )
    model.eval()
    return model


def find_masked_positions(input_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Return a [num_masked, 2] LongTensor of (row, col) indices holding ``mask_token_id``.

    Works for a single sequence [seq] or a batch [batch, seq]; a 1-D input is treated as one
    row so column indices are always meaningful.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    positions = (input_ids == mask_token_id).nonzero(as_tuple=False)
    return positions


def apply_mask_at(
    input_ids: torch.Tensor,
    positions,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace ``positions`` (column indices) of a single sequence with ``mask_token_id``.

    Returns ``(masked_input, original_tokens)`` where ``original_tokens`` holds the ids that
    were overwritten (in the given order), so callers can score predictions against truth.
    ``input_ids`` must be a single sequence [seq]; a copy is returned (input is not mutated).
    """
    if input_ids.dim() != 1:
        raise ValueError("apply_mask_at expects a single sequence of shape [seq].")
    positions = list(positions)
    masked = input_ids.clone()
    originals = input_ids[positions].clone()
    masked[positions] = mask_token_id
    return masked, originals


@torch.no_grad()
def topk_predictions(
    logits: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k token ids and probabilities from logits.

    ``logits`` has shape [..., vocab]; returns ``(ids, probs)`` each of shape [..., k]. Softmax
    is taken over the vocabulary dimension so probabilities are comparable across positions.
    """
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = probs.topk(k, dim=-1)
    return top_ids, top_probs


@torch.no_grad()
def predict_masked_topk(
    model: BertForMaskedLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    k: int = 5,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the model and return top-k predictions at every masked position.

    Returns ``(positions, ids, probs)`` where ``positions`` is [num_masked, 2] (row, col),
    ``ids`` is [num_masked, k], and ``probs`` is [num_masked, k].
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    model.eval()
    logits = model(input_ids, attention_mask=attention_mask)["logits"]  # [b, seq, vocab]
    positions = find_masked_positions(input_ids, mask_token_id)  # [num_masked, 2]
    if positions.numel() == 0:
        empty = torch.empty(0, k, dtype=torch.long)
        return positions, empty, empty.float()
    sel = logits[positions[:, 0], positions[:, 1]]  # [num_masked, vocab]
    ids, probs = topk_predictions(sel, k)
    return positions, ids, probs


@torch.no_grad()
def masked_accuracy_topk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ks=(1, 5),
) -> dict[int, float]:
    """Top-k masked-token accuracy given full logits [b, seq, vocab] and labels [b, seq].

    Positions with label == -100 are ignored. Returns {k: accuracy} for each k in ``ks``.
    """
    mask = labels != -100
    if int(mask.sum()) == 0:
        return {k: 0.0 for k in ks}
    sel_logits = logits[mask]  # [num_masked, vocab]
    sel_labels = labels[mask]  # [num_masked]
    max_k = max(ks)
    top = sel_logits.topk(max_k, dim=-1).indices  # [num_masked, max_k]
    correct = top == sel_labels.unsqueeze(-1)
    out = {}
    for k in ks:
        out[k] = float(correct[:, :k].any(dim=-1).float().mean())
    return out
