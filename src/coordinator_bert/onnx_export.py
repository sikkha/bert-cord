"""ONNX export + portable-inference helpers for the custom ``BertForMaskedLM`` (Milestone 0.7).

Scope: **masked-token prediction only.** The exported graph computes MLM logits from
``input_ids`` / ``attention_mask`` / ``token_type_ids`` — nothing else. It contains the
inference computation graph and the trained weights, with dynamic batch and sequence
dimensions. It deliberately does **not** contain optimizer/scheduler/RNG state, the training
loss, labels, checkpoint-manager logic, or dict / variable-length attention-probability
outputs. PyTorch checkpoints remain the training source of truth; the ONNX file is a portable
inference artifact.

The model architecture is not modified: ``MLMInferenceWrapper`` only adapts the call signature
(single tensor in → single tensor out) so the graph has a fixed, clean I/O contract.

ONNX / ONNX Runtime are optional dependencies — they are imported lazily with an actionable
error if missing (install ``.[onnx]``).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn

from .configuration import ModelConfig
from .inference import load_model_for_inference
from .model import BertForMaskedLM, count_parameters

# Fixed I/O contract for the exported graph.
INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")
OUTPUT_NAMES = ("logits",)
# Default opset: torch 2.13's ONNX exporter implements opset 18 natively; onnx>=1.22 and
# onnxruntime>=1.16 support it. Chosen to avoid a lossy down-conversion (see docs/ONNX_EXPORT.md).
DEFAULT_OPSET = 18


class OnnxDependencyError(RuntimeError):
    """Raised when an ONNX/ONNX Runtime package is required but not installed."""


def _import_onnx():
    try:
        import onnx  # noqa: PLC0415
        return onnx
    except Exception as e:  # noqa: BLE001
        raise OnnxDependencyError(
            "The 'onnx' package is required for this operation. Install with: "
            "python -m pip install -e \".[onnx]\""
        ) from e


def _import_ort():
    try:
        import onnxruntime as ort  # noqa: PLC0415
        return ort
    except Exception as e:  # noqa: BLE001
        raise OnnxDependencyError(
            "The 'onnxruntime' package is required for this operation. Install with: "
            "python -m pip install -e \".[onnx]\""
        ) from e


class MLMInferenceWrapper(nn.Module):
    """Inference-only adapter: (input_ids, attention_mask, token_type_ids) -> logits tensor.

    Calls the underlying model with ``labels=None`` and ``return_probs=False`` so the graph
    has no loss branch and no variable-length attention-probability output. Does not alter the
    wrapped module's parameters or architecture.
    """

    def __init__(self, model: BertForMaskedLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None,
            return_probs=False,
        )
        return out["logits"]


def build_inference_wrapper(model: BertForMaskedLM) -> MLMInferenceWrapper:
    """Wrap a model for inference-only export and put it in eval mode."""
    model.eval()
    return MLMInferenceWrapper(model).eval()


def example_inputs(batch: int, seq: int, vocab_size: int, pad_token_id: int = 0,
                   seed: int = 0, pad_last: int = 0):
    """Deterministic (input_ids, attention_mask, token_type_ids) for export/tests.

    ``pad_last`` optionally marks the final ``pad_last`` columns as padding (attention_mask 0)
    so mask behavior can be exercised. All tensors are int64, as ONNX expects for indices.
    """
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch, seq), generator=g, dtype=torch.long)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    if pad_last > 0:
        attention_mask[:, seq - pad_last:] = 0
        input_ids[:, seq - pad_last:] = pad_token_id
    token_type_ids = torch.zeros(batch, seq, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids


def export_to_onnx(
    model: BertForMaskedLM,
    output_path: str,
    *,
    sequence_length: int = 128,
    opset: int = DEFAULT_OPSET,
    batch_size: int = 1,
    dynamic: bool = True,
) -> dict:
    """Export ``model`` (as an inference wrapper) to ONNX with dynamic batch & seq axes.

    Returns a metadata dict (path, size_bytes, opset, input/output names, param count).
    Raises OnnxDependencyError if onnx is missing; other errors propagate to the caller.
    """
    onnx = _import_onnx()  # fail fast + validate afterwards
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    wrapper = build_inference_wrapper(model)
    vocab = model.config.vocab_size
    pad = model.config.pad_token_id
    ii, am, tt = example_inputs(batch_size, sequence_length, vocab, pad_token_id=pad)

    dynamic_axes = None
    if dynamic:
        axes = {0: "batch", 1: "sequence"}
        dynamic_axes = {name: dict(axes) for name in INPUT_NAMES}
        dynamic_axes["logits"] = {0: "batch", 1: "sequence"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (ii, am, tt),
            output_path,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    # Structural validation immediately after export (load_external_data resolves any
    # sibling .onnx.data weight file the exporter may have written).
    onnx.checker.check_model(onnx.load(output_path))

    graph_size = os.path.getsize(output_path)
    external = _external_data_files(output_path)
    external_bytes = sum(os.path.getsize(p) for p in external)
    return {
        "output_path": output_path,
        "graph_size_bytes": graph_size,
        "external_data_files": external,
        "external_data_bytes": external_bytes,
        "size_bytes": graph_size + external_bytes,  # total on-disk artifact size
        "opset": opset,
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "dynamic": dynamic,
        "sequence_length": sequence_length,
        "vocab_size": vocab,
        "param_count": count_parameters(model)["unique"],
    }


def _external_data_files(output_path: str) -> list:
    """Return sibling external-weight files (e.g. <model>.onnx.data) written next to the graph.

    The torch ONNX exporter may store weights in an external data file so the .onnx graph stays
    small. Both files must be shipped together for ONNX Runtime to load the model.
    """
    candidates = [output_path + ".data"]
    base = os.path.splitext(output_path)[0]
    candidates.append(base + ".data")
    return sorted({c for c in candidates if os.path.exists(c) and c != output_path})


def export_checkpoint_to_onnx(config: ModelConfig, checkpoint_path: Optional[str],
                              output_path: str, **kwargs) -> dict:
    """Load a checkpoint on CPU (via the existing resolution logic) and export it to ONNX."""
    model = load_model_for_inference(config, checkpoint_path, map_location="cpu")
    meta = export_to_onnx(model, output_path, **kwargs)
    meta["checkpoint"] = checkpoint_path
    return meta


# --------------------------------------------------------------------------------------- #
# ONNX Runtime helpers (CPU by default)
# --------------------------------------------------------------------------------------- #
def check_onnx_model(path: str) -> None:
    """Run ONNX structural validation (raises on failure)."""
    onnx = _import_onnx()
    onnx.checker.check_model(onnx.load(path))


def create_ort_session(path: str, providers: Optional[list] = None):
    """Create an ONNX Runtime session (CPU by default)."""
    ort = _import_ort()
    providers = providers or ["CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=providers)


def ort_input_names(session) -> list:
    return [i.name for i in session.get_inputs()]


def run_onnx_logits(session, input_ids, attention_mask, token_type_ids):
    """Run the session and return the logits ndarray. Inputs are torch tensors or ndarrays."""
    import numpy as np

    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().astype(np.int64)
        return np.asarray(x).astype(np.int64)

    feed = {"input_ids": _np(input_ids), "attention_mask": _np(attention_mask),
            "token_type_ids": _np(token_type_ids)}
    names = set(ort_input_names(session))
    missing = names - set(feed)
    if missing:
        raise ValueError(f"session expects inputs {sorted(names)}; missing {sorted(missing)}")
    feed = {k: v for k, v in feed.items() if k in names}
    return session.run(["logits"], feed)[0]


def torch_reference_logits(model: BertForMaskedLM, input_ids, attention_mask, token_type_ids):
    """Reference PyTorch logits (numpy) from the inference wrapper, eval + no_grad."""
    wrapper = build_inference_wrapper(model)
    with torch.no_grad():
        out = wrapper(input_ids, attention_mask, token_type_ids)
    return out.cpu().numpy()


def compare_logits(a, b) -> dict:
    """Numerical comparison stats between two logits arrays (max/mean abs diff, allclose)."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    return {
        "shape_a": tuple(a.shape),
        "shape_b": tuple(b.shape),
        "shapes_match": a.shape == b.shape,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "any_nan": bool(np.isnan(a).any() or np.isnan(b).any()),
        "any_inf": bool(np.isinf(a).any() or np.isinf(b).any()),
    }


def topk_agreement(a, b, positions, k: int = 5) -> float:
    """Fraction of (row, col) positions where the top-k token ids agree (order-insensitive)."""
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    if not positions:
        return 1.0
    agree = 0
    for (r, c) in positions:
        ta = set(np.argsort(-a[r, c])[:k].tolist())
        tb = set(np.argsort(-b[r, c])[:k].tolist())
        agree += 1 if ta == tb else 0
    return agree / len(positions)
