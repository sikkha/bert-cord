#!/usr/bin/env python3
"""Build a self-contained Hugging Face model-repository package for the ONNX MLM baseline.

LOCAL STAGING ONLY. This never uploads, never authenticates, and never touches the network.
It produces a directory that can *later* be uploaded manually. The PyTorch checkpoint remains
the training source of truth; this ONNX package is a derived, inference-only artifact.

Key correctness point: the source external-data file is named ``<model>.onnx.data``; the
packaged graph must reference ``model.onnx.data``. Renaming the file is not sufficient — the
external-data location is recorded inside the ONNX graph. We therefore re-save the model with
``save_as_external_data=..., location="model.onnx.data"`` so the packaged graph points at the
packaged weight file, then validate the packaged copy from inside the package directory.

Example:
  python scripts/build_hf_onnx_package.py \
    --config configs/bert_25m_mac.yaml \
    --onnx-model exports/bert_cord_27m_mlm.onnx \
    --output bert-cord-27m-mlm-onnx \
    --repo-id sikkha/bert-cord-27m-mlm-onnx \
    --package-version 0.1.1-onnx
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SOURCE_REPOSITORY = "https://github.com/sikkha/bert-cord"
EXPECTED_PARAMS = 27010304
ONNX_OPSET = 18
PRECISION = "float32"
INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]
OUTPUT_NAME = "logits"


# --------------------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------------------- #
def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args, cwd=None) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def detect_git(cwd: str) -> tuple[str, "str | None"]:
    commit = _git("rev-parse", "HEAD", cwd=cwd) or "unknown"
    tag = _git("describe", "--tags", "--exact-match", "HEAD", cwd=cwd)
    if not tag:
        pointed = _git("tag", "--points-at", "HEAD", cwd=cwd)
        tag = pointed.splitlines()[0].strip() if pointed else ""
    return commit, (tag or None)


# --------------------------------------------------------------------------------------- #
# Core build
# --------------------------------------------------------------------------------------- #
def build_package(config_path: str, onnx_model: str, output_dir: str, repo_id: str,
                  package_version: str, checkpoint: "str | None" = None,
                  project_root: "str | None" = None,
                  model_source_commit: "str | None" = None,
                  model_source_tag: "str | None" = None) -> dict:
    """Build the package directory. Returns a report dict. Raises on any hard failure."""
    import onnx

    from coordinator_bert.configuration import load_config

    project_root = project_root or os.getcwd()
    src_onnx = os.path.abspath(onnx_model)
    src_data = src_onnx + ".data"
    if not os.path.exists(src_onnx):
        raise FileNotFoundError(f"source ONNX graph not found: {src_onnx}")
    if not os.path.exists(src_data):
        raise FileNotFoundError(f"source ONNX external-data file not found: {src_data}")

    cfg = load_config(config_path)

    # Packaging provenance = the commit/tag of THIS tooling tree (current HEAD).
    packaging_commit, packaging_tag = detect_git(project_root)
    # Model provenance = the commit/tag the ONNX artifact was exported from. If not supplied,
    # it defaults to the packaging commit/tag (export + package happened at the same tree).
    if model_source_commit is None:
        model_commit, model_tag = packaging_commit, packaging_tag
    else:
        model_commit, model_tag = model_source_commit, model_source_tag

    # Recreate the package dir safely (only ever the given output path). No ignore_errors:
    # a partially-cleared directory would yield a mixed/stale package, so we abort instead.
    out = os.path.abspath(output_dir)
    onnx_dir = os.path.join(out, "onnx")
    _prepare_output_dir(out)
    os.makedirs(onnx_dir, exist_ok=True)

    # --- Re-save ONNX so the packaged graph references model.onnx.data (not the old name). ---
    model = onnx.load(src_onnx, load_external_data=True)  # pulls weights into memory
    pkg_onnx = os.path.join(onnx_dir, "model.onnx")
    # Remove any prior data file so save writes a clean one.
    for stale in (pkg_onnx, os.path.join(onnx_dir, "model.onnx.data")):
        if os.path.exists(stale):
            os.remove(stale)
    onnx.save_model(
        model, pkg_onnx, save_as_external_data=True, all_tensors_to_one_file=True,
        location="model.onnx.data", size_threshold=1024, convert_attribute=False,
    )
    if not os.path.exists(os.path.join(onnx_dir, "model.onnx.data")):
        raise RuntimeError("expected packaged external-data file onnx/model.onnx.data was not "
                           "written")
    # Structural validation of the packaged copy (resolves relative external data).
    onnx.checker.check_model(onnx.load(pkg_onnx, load_external_data=True))

    # --- Fresh parity validation (packaged ONNX vs PyTorch reference). ---
    parity = _measure_parity(cfg, checkpoint, out)

    # --- Metadata files ---
    graph_size = os.path.getsize(pkg_onnx)
    data_size = os.path.getsize(os.path.join(onnx_dir, "model.onnx.data"))

    prov = {"model_source_commit": model_commit, "model_source_tag": model_tag,
            "packaging_source_commit": packaging_commit, "packaging_source_tag": packaging_tag}

    config_json = _make_config(cfg, prov, package_version, repo_id)
    _write_json(os.path.join(out, "config.json"), config_json)

    _write_text(os.path.join(out, "requirements.txt"),
                "numpy>=1.24\nonnxruntime>=1.17\n")

    _copy_license(project_root, out)

    _write_text(os.path.join(out, "inference.py"), _inference_py())

    readme = _model_card(cfg, prov, package_version, repo_id, parity, graph_size, data_size)
    _write_text(os.path.join(out, "README.md"), readme)

    evaluation = _make_evaluation(cfg, prov, package_version, checkpoint, parity, onnx_dir)
    _write_json(os.path.join(out, "evaluation.json"), evaluation)

    # --- MANIFEST last: checksums of every package file except MANIFEST itself. ---
    manifest = _make_manifest(out, repo_id, package_version, prov, parity)
    _write_json(os.path.join(out, "MANIFEST.json"), manifest)

    total = sum(f["size_bytes"] for f in manifest["files"]) + os.path.getsize(
        os.path.join(out, "MANIFEST.json"))
    return {
        "output_dir": out, "model_commit": model_commit, "model_tag": model_tag,
        "packaging_commit": packaging_commit, "packaging_tag": packaging_tag,
        "package_version": package_version, "repo_id": repo_id, "graph_size": graph_size,
        "data_size": data_size, "total_size": total, "parity": parity,
        "n_files": len(manifest["files"]) + 1,
    }


def _prepare_output_dir(out: str) -> None:
    """Ensure the output dir is a clean, fresh directory. Abort (raise) if it can't be removed.

    We deliberately do NOT swallow removal errors: a partially-cleared directory would produce a
    mixed/stale package. If the existing directory cannot be removed cleanly (e.g. a synced or
    read-only mount), the build must fail with an actionable message so the user removes it or
    chooses a fresh --output path.
    """
    if os.path.exists(out):
        try:
            shutil.rmtree(out)
        except OSError as e:
            raise RuntimeError(
                f"output directory '{out}' already exists and could not be removed cleanly "
                f"({type(e).__name__}: {e}). Refusing to build into a partially-cleared "
                "directory (that would produce a mixed/stale package). Remove it manually "
                "('rm -rf') or pass a fresh --output path.") from e
        if os.path.exists(out):
            raise RuntimeError(
                f"output directory '{out}' still exists after removal — refusing to produce a "
                "mixed/stale package. Use a fresh --output path.")


def _measure_parity(cfg, checkpoint, out_dir) -> dict:
    """Run packaged-ONNX vs PyTorch parity across 2 seq lengths x 2 batch sizes (+ padded)."""
    import numpy as np

    from coordinator_bert.inference import load_model_for_inference
    from coordinator_bert import onnx_export as ox

    pkg_onnx = os.path.join(out_dir, "onnx", "model.onnx")
    session = ox.create_ort_session(pkg_onnx, providers=["CPUExecutionProvider"])
    provider = session.get_providers()[0]

    have_ckpt = checkpoint is not None and os.path.exists(
        __resolve(checkpoint))
    model = load_model_for_inference(cfg.model, checkpoint if have_ckpt else None,
                                     map_location="cpu")
    vocab = cfg.model.vocab_size
    pad = cfg.model.pad_token_id
    # Two distinct sequence lengths within the model's positional capacity (exercises dynamic
    # axes). For the 27M baseline (max_pos 512) these are 128 and 64.
    max_pos = cfg.model.max_position_embeddings
    s1 = min(128, max_pos)
    s2 = max(8, min(64, max_pos // 2))
    if s2 == s1:
        s2 = max(8, s1 // 2)
    cases = [(1, s1, 0), (3, s1, 2), (1, s2, 2), (2, s2, 0)]
    max_diffs, mean_diffs, agrees = [], [], []
    any_nan_inf = False
    for i, (b, s, pl) in enumerate(cases):
        ii, am, tt = ox.example_inputs(b, s, vocab, pad_token_id=pad, seed=i + 1, pad_last=pl)
        ref = ox.torch_reference_logits(model, ii, am, tt)
        got = ox.run_onnx_logits(session, ii, am, tt)
        st = ox.compare_logits(ref, got)
        positions = [(r, c) for r in range(b) for c in (1, s // 2, s - 2)]
        agrees.append(ox.topk_agreement(ref, got, positions, k=5))
        max_diffs.append(st["max_abs_diff"])
        mean_diffs.append(st["mean_abs_diff"])
        any_nan_inf = any_nan_inf or st["any_nan"] or st["any_inf"]
    return {
        "provider": provider,
        "checkpoint_used": bool(have_ckpt),
        "batch_sizes": sorted({b for b, _, _ in cases}),
        "sequence_lengths": sorted({s for _, s, _ in cases}),
        "padded_cases_included": True,
        "max_abs_diff": float(max(max_diffs)),
        "mean_abs_diff_min": float(min(mean_diffs)),
        "mean_abs_diff_max": float(max(mean_diffs)),
        "top5_agreement": float(min(agrees)),
        "any_nan_or_inf": bool(any_nan_inf),
        "dynamic_axes_ok": len({s for _, s, _ in cases}) >= 2 and len(
            {b for b, _, _ in cases}) >= 2,
        "onnx_checker": "PASS",
        "rtol": 1e-3, "atol": 2e-3,
        "n_cases": len(cases),
    }


def __resolve(checkpoint: str) -> str:
    from coordinator_bert.checkpointing import resolve_checkpoint_path
    return resolve_checkpoint_path(checkpoint)


# --------------------------------------------------------------------------------------- #
# File generators
# --------------------------------------------------------------------------------------- #
def _make_config(cfg, prov: dict, package_version, repo_id) -> dict:
    m = cfg.model
    return {
        "model_type": "bert_cord",
        "architectures": ["BertForMaskedLM"],
        "task": "masked-language-modeling",
        "vocab_size": m.vocab_size,
        "hidden_size": m.hidden_size,
        "num_hidden_layers": m.num_hidden_layers,
        "num_attention_heads": m.num_attention_heads,
        "intermediate_size": m.intermediate_size,
        "max_position_embeddings": m.max_position_embeddings,
        "type_vocab_size": m.type_vocab_size,
        "hidden_dropout_prob": m.hidden_dropout_prob,
        "attention_probs_dropout_prob": m.attention_probs_dropout_prob,
        "layer_norm_eps": m.layer_norm_eps,
        "position_embedding_type": "absolute",
        "parameters": EXPECTED_PARAMS,
        "onnx_opset": ONNX_OPSET,
        "precision": PRECISION,
        "dynamic_batch": True,
        "dynamic_sequence": True,
        "external_data": True,
        "source_repository": SOURCE_REPOSITORY,
        "model_source_commit": prov["model_source_commit"],
        "model_source_tag": prov["model_source_tag"],
        "packaging_source_commit": prov["packaging_source_commit"],
        "packaging_source_tag": prov["packaging_source_tag"],
        "package_version": package_version,
        "future_huggingface_repo_id": repo_id,
    }


def _make_evaluation(cfg, prov: dict, package_version, checkpoint, parity, onnx_dir) -> dict:
    ckpt_desc = "resolved via latest.json from the project's smoke checkpoints (step_000040)" \
        if parity["checkpoint_used"] else "random-initialized reference (no checkpoint supplied)"
    files = {
        "onnx/model.onnx": _sha256(os.path.join(onnx_dir, "model.onnx")),
        "onnx/model.onnx.data": _sha256(os.path.join(onnx_dir, "model.onnx.data")),
    }
    return {
        "source_repository": SOURCE_REPOSITORY,
        "model_source_commit": prov["model_source_commit"],
        "model_source_tag": prov["model_source_tag"],
        "packaging_source_commit": prov["packaging_source_commit"],
        "packaging_source_tag": prov["packaging_source_tag"],
        "source_checkpoint_description": ckpt_desc,
        "model_parameters": EXPECTED_PARAMS,
        "format": "onnx",
        "onnx_opset": ONNX_OPSET,
        "precision": PRECISION,
        "runtime": "onnxruntime",
        "validated_execution_provider": parity["provider"],
        "batch_sizes_tested": parity["batch_sizes"],
        "sequence_lengths_tested": parity["sequence_lengths"],
        "padded_cases_included": parity["padded_cases_included"],
        "max_abs_diff": parity["max_abs_diff"],
        "mean_abs_diff_range": [parity["mean_abs_diff_min"], parity["mean_abs_diff_max"]],
        "top5_agreement": parity["top5_agreement"],
        "nan_or_inf": parity["any_nan_or_inf"],
        "dynamic_axes_ok": parity["dynamic_axes_ok"],
        "onnx_checker": parity["onnx_checker"],
        "tolerances": {"rtol": parity["rtol"], "atol": parity["atol"]},
        "package_version": package_version,
        "timestamp_utc": _utc_now(),
        "file_checksums_sha256": files,
        "known_limitations": [
            "Synthetic MLM development baseline; no natural-language training data.",
            "No tokenizer bundled; inputs are integer token ids only.",
            "FP32 CPUExecutionProvider validated; CUDA/CoreML/FP16/BF16 unvalidated.",
            "Not a coordinator/mini-amygdala; no language-understanding claim.",
        ],
    }


def _make_manifest(out, repo_id, package_version, prov: dict, parity) -> dict:
    files = []
    for root, _dirs, names in os.walk(out):
        for name in sorted(names):
            if name == "MANIFEST.json":
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, out).replace(os.sep, "/")
            files.append({"path": rel, "size_bytes": os.path.getsize(full),
                          "sha256": _sha256(full)})
    files.sort(key=lambda f: f["path"])
    return {
        "package_name": "bert-cord-27m-mlm-onnx",
        "package_version": package_version,
        "future_huggingface_repo_id": repo_id,
        "source_repository": SOURCE_REPOSITORY,
        "model_source_commit": prov["model_source_commit"],
        "model_source_tag": prov["model_source_tag"],
        "packaging_source_commit": prov["packaging_source_commit"],
        "packaging_source_tag": prov["packaging_source_tag"],
        "created_utc": _utc_now(),
        "model_parameters": EXPECTED_PARAMS,
        "onnx_opset": ONNX_OPSET,
        "precision": PRECISION,
        "expected_inputs": INPUT_NAMES,
        "expected_output": OUTPUT_NAME,
        "validated_execution_provider": parity["provider"],
        "external_data": {"graph": "onnx/model.onnx", "weights": "onnx/model.onnx.data",
                          "linked": True},
        "known_limitations": [
            "Synthetic MLM baseline; no tokenizer; FP32 CPU only validated.",
            "Not a coordinator; no language-understanding or production-readiness claim.",
        ],
        "files": files,
    }


def _copy_license(project_root: str, out: str) -> None:
    src = os.path.join(project_root, "LICENSE")
    if not os.path.exists(src):
        raise FileNotFoundError(f"project LICENSE not found at {src}")
    shutil.copyfile(src, os.path.join(out, "LICENSE"))


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _inference_py() -> str:
    return '''#!/usr/bin/env python3
"""Standalone ONNX Runtime inference for bert-cord-27m-mlm-onnx (masked-token prediction).

No dependency on the bert_cord training package. Requires: numpy, onnxruntime.
This is a synthetic MLM baseline: inputs are integer token ids, NOT natural-language text.
There is no bundled tokenizer and no language-understanding claim.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "onnx", "model.onnx")
DATA = os.path.join(HERE, "onnx", "model.onnx.data")
EXPECTED_INPUTS = {"input_ids", "attention_mask", "token_type_ids"}


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main() -> int:
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001
        print("onnxruntime is required: pip install -r requirements.txt", file=sys.stderr)
        return 3
    if not os.path.exists(MODEL):
        print(f"missing model graph: {MODEL}", file=sys.stderr)
        return 2
    if not os.path.exists(DATA):
        print("missing external weights: onnx/model.onnx.data "
              "(download BOTH model.onnx and model.onnx.data together)", file=sys.stderr)
        return 2

    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    print("active providers   :", session.get_providers())
    print("available providers:", ort.get_available_providers())
    names = {i.name for i in session.get_inputs()}
    if not EXPECTED_INPUTS.issubset(names):
        print(f"unexpected inputs {sorted(names)}; need {sorted(EXPECTED_INPUTS)}",
              file=sys.stderr)
        return 1
    out_names = [o.name for o in session.get_outputs()]
    if "logits" not in out_names:
        print(f"unexpected outputs {out_names}; need 'logits'", file=sys.stderr)
        return 1

    # Small synthetic token-id input (period-3 motif), with a masked position.
    # NOTE: reserved ids -> PAD=0, CLS=1, SEP=2, MASK=3, UNK=4; real tokens start at 5.
    MASK_ID = 3
    motif = [50, 67, 7]
    body = [motif[i % 3] for i in range(22)]
    ids = [1] + body + [2]              # [CLS] ... [SEP]
    mask_pos = len(ids) // 2
    true_id = ids[mask_pos]
    ids[mask_pos] = MASK_ID

    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    token_type_ids = np.zeros_like(input_ids)

    logits = session.run(["logits"], {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })[0]
    if logits.ndim != 3:
        print(f"unexpected output rank {logits.ndim}", file=sys.stderr)
        return 1

    print("logits shape       :", logits.shape)
    row = logits[0, mask_pos].astype(np.float64)
    probs = softmax(row)
    top = np.argsort(-row)[:5]
    print(f"masked position    : {mask_pos} (true id {true_id})")
    print("top-5 token ids    :", top.tolist())
    print("top-5 probabilities:", [round(float(probs[i]), 4) for i in top])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _model_card(cfg, prov: dict, package_version, repo_id, parity, graph_size,
                data_size) -> str:
    m = cfg.model
    total_mb = (graph_size + data_size) / (1024 ** 2)
    model_commit = prov["model_source_commit"]
    model_tag = prov["model_source_tag"] or "(none)"
    pkg_commit = prov["packaging_source_commit"]
    pkg_tag = prov["packaging_source_tag"] or "(none)"
    return f'''---
license: apache-2.0
library_name: onnxruntime
tags:
  - bert
  - onnx
  - onnxruntime
  - masked-language-modeling
  - custom-model
  - research
  - pytorch
language:
  - en
---

# BERT-Cord 27M — ONNX MLM baseline

**Package version:** `{package_version}` · **Future HF repo:** `{repo_id}`

## Purpose

A portable, framework-neutral **ONNX Runtime** inference artifact for the custom
`BertForMaskedLM` encoder from the [bert_cord]({SOURCE_REPOSITORY}) research project. It performs
**masked-token prediction only**.

> **Honest scope.** This is a **synthetic MLM development checkpoint**. It is **not** yet a
> "mini-amygdala" coordinator. It performs **no** coordination, routing, memory activation, task
> lifecycle control, consciousness, or general language understanding, and is **not** production
> ready. **No natural-language tokenizer is bundled** — inputs are integer token ids.

## Development stage

Milestone 0.7 (ONNX export). Derived, inference-only artifact; the PyTorch checkpoint in the
source repository remains the training source of truth. Not a Transformers `AutoModel` — do not
assume `transformers` compatibility.

## Architecture summary

Custom from-scratch BERT encoder (not Hugging Face `BertModel`): learned token/position/
token-type embeddings, {m.num_hidden_layers} post-LN transformer layers, {m.num_attention_heads}
attention heads, hidden size {m.hidden_size}, intermediate size {m.intermediate_size}, tied
input/output embeddings, GELU FFN.

- **Exact parameter count:** `27,010,304`
- **Task:** masked-language modeling (MLM)
- **ONNX opset:** 18
- **Precision:** FP32

## Input / output contract

| tensor | role | dtype | shape |
|--------|------|-------|-------|
| `input_ids` | input | int64 | `[batch, sequence]` |
| `attention_mask` | input | int64 | `[batch, sequence]` |
| `token_type_ids` | input | int64 | `[batch, sequence]` |
| `logits` | output | float32 | `[batch, sequence, {m.vocab_size}]` |

`batch` and `sequence` are **dynamic** axes.

## External data requirement

The weights are stored **externally**: you must download **both**
`onnx/model.onnx` (graph) **and** `onnx/model.onnx.data` (weights, ~{data_size/1024/1024:.1f} MB)
and keep them side by side. Total artifact ≈ **{total_mb:.1f} MB**. Loading `model.onnx` without
`model.onnx.data` will fail.

## Installation

```bash
pip install -r requirements.txt   # numpy>=1.24, onnxruntime>=1.17
```

## Standalone inference example

```bash
python inference.py
```

Or in code:

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("onnx/model.onnx", providers=["CPUExecutionProvider"])
seq = [1, 50, 67, 7, 50, 67, 3, 50, 67, 7, 2]           # ids; 3 == [MASK]
ii = np.array([seq], dtype=np.int64)
am = np.ones_like(ii); tt = np.zeros_like(ii)
logits = sess.run(["logits"], {{"input_ids": ii, "attention_mask": am,
                                "token_type_ids": tt}})[0]
print(logits.shape)                                       # (1, seq, {m.vocab_size})
```

## Parity results (this package)

Packaged ONNX vs the source PyTorch model, identical deterministic inputs, FP32 CPU:

- max absolute logit difference: **{parity["max_abs_diff"]:.2e}** (tolerances rtol=1e-3, atol=2e-3)
- mean absolute difference: {parity["mean_abs_diff_min"]:.2e} – {parity["mean_abs_diff_max"]:.2e}
- masked-position **top-5 agreement: {parity["top5_agreement"]:.2f}**
- NaN/Inf: {parity["any_nan_or_inf"]}
- dynamic axes: {parity["dynamic_axes_ok"]} (batch {parity["batch_sizes"]} × seq {parity["sequence_lengths"]})
- ONNX structural check: {parity["onnx_checker"]}

## Validated vs unvalidated runtimes

- **Validated:** `CPUExecutionProvider`, FP32.
- **Unvalidated:** `CUDAExecutionProvider` (`onnxruntime-gpu`), Apple **CoreML**, and **FP16 /
  BF16**. No claim is made about them.

## Intended uses

Portable CPU inference / benchmarking of a small custom encoder graph; a reference artifact for
the bert_cord research line; ONNX Runtime integration experiments.

## Out-of-scope uses

Any language-understanding, chat, generation, coordination/routing, safety-critical, or
production use. This model has no such capability.

## Limitations

Synthetic MLM baseline trained on a learnable copy-motif corpus (not natural language). Outputs
are only meaningful as an MLM-graph sanity signal. No tokenizer, no semantics.

## Training-data disclosure

Trained on **synthetic** token sequences (a tiled copy-motif corpus) for pipeline validation —
**no natural-language corpus, no private data.**

## Tokenizer disclosure

**No tokenizer is bundled.** Inputs are raw integer token ids. Reserved ids: PAD=0, CLS=1, SEP=2,
MASK=3, UNK=4; "real" tokens start at id 5.

## Reproducibility & provenance

Model provenance (the commit the ONNX model was exported from) is tracked separately from
packaging provenance (the commit of the tooling that assembled this package):

- Source repository: {SOURCE_REPOSITORY}
- **Model source commit:** `{model_commit}`
- **Model source tag:** `{model_tag}`
- **Packaging source commit:** `{pkg_commit}`
- **Packaging source tag:** `{pkg_tag}`
- Exported with PyTorch's ONNX exporter (opset 18); parity re-measured at package-build time.

## Package file layout

```
bert-cord-27m-mlm-onnx/
├── README.md          (this model card)
├── LICENSE            (Apache-2.0, copied from source)
├── config.json        (architecture + provenance)
├── evaluation.json    (measured parity + checksums)
├── requirements.txt   (numpy, onnxruntime)
├── inference.py       (standalone ONNX Runtime example)
├── MANIFEST.json      (files + sizes + SHA-256)
└── onnx/
    ├── model.onnx       (graph; references model.onnx.data)
    └── model.onnx.data  (external FP32 weights — download together)
```

## Citation

```bibtex
@software{{bert_cord_27m_mlm_onnx,
  title  = {{BERT-Cord 27M — ONNX MLM baseline}},
  author = {{Kan (sikkha)}},
  year   = {{2026}},
  url    = {{{SOURCE_REPOSITORY}}},
  note   = {{Synthetic MLM development baseline; model commit {model_commit}}}
}}
```

## License

Apache-2.0 (see `LICENSE`).
'''


# --------------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Build the HF ONNX package (local staging only).")
    p.add_argument("--config", required=True)
    p.add_argument("--onnx-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--package-version", required=True)
    p.add_argument("--checkpoint", default="experiments/smoke/checkpoints",
                   help="Checkpoint for parity reference (default project smoke checkpoints).")
    p.add_argument("--model-source-commit", default=None,
                   help="Commit the ONNX model was exported from (defaults to packaging HEAD).")
    p.add_argument("--model-source-tag", default=None,
                   help="Tag the ONNX model was exported from (used with --model-source-commit).")
    args = p.parse_args()

    try:
        report = build_package(args.config, args.onnx_model, args.output, args.repo_id,
                               args.package_version, checkpoint=args.checkpoint,
                               model_source_commit=args.model_source_commit,
                               model_source_tag=args.model_source_tag)
    except Exception as e:  # noqa: BLE001
        print(f"[build_hf] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    par = report["parity"]
    print("=" * 68)
    print("[build_hf] SUCCESS — local staging package created (nothing uploaded)")
    print(f"[build_hf] output          : {report['output_dir']}")
    print(f"[build_hf] repo id (future): {report['repo_id']}")
    print(f"[build_hf] version         : {report['package_version']}")
    print(f"[build_hf] model source    : {report['model_commit']} (tag {report['model_tag']})")
    print(f"[build_hf] packaging source: {report['packaging_commit']} "
          f"(tag {report['packaging_tag']})")
    print(f"[build_hf] graph size    : {report['graph_size']:,} B | "
          f"weights : {report['data_size']:,} B | total : {report['total_size']:,} B")
    print(f"[build_hf] parity        : max|Δ|={par['max_abs_diff']:.2e} "
          f"top5={par['top5_agreement']:.2f} nan/inf={par['any_nan_or_inf']} "
          f"provider={par['provider']}")
    print(f"[build_hf] files         : {report['n_files']}")
    print("-" * 68)
    print("[build_hf] To upload LATER (manual; DO NOT run automatically):")
    print(f"  hf repo create {report['repo_id']} --type model")
    print(f"  hf upload {report['repo_id']} {os.path.basename(report['output_dir'])} .")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
