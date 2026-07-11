# ONNX export & portable inference (Milestone 0.7)

`bert_cord` can export the trained `BertForMaskedLM` to a portable **ONNX** artifact and run it
with **ONNX Runtime** (CPU by default). Scope is **masked-token prediction only** — no
coordinator routing, memory activation, task-lifecycle control, or language understanding.

## What ONNX solves

A framework-neutral, self-contained inference graph + weights that runs without the training
stack (no Accelerate, no optimizer, no checkpoint-manager). It enables portable CPU inference
now and a path to CUDA/other execution providers later, decoupled from the PyTorch training
environment.

### Why ONNX Runtime (not vLLM / Ollama / llama.cpp)

Those tools target **decoder-only autoregressive LLMs** (KV-cache generation, GGUF, chat
serving). `bert_cord` is an **encoder-only MLM** producing per-position logits in a single
forward pass — there is no token-by-token generation to accelerate. ONNX Runtime is the right
fit for a small encoder: it runs the exact graph on CPU/GPU with minimal dependencies and gives
verifiable numerical parity with PyTorch.

## What is included / excluded

Included: the inference computation graph, the trained weights, dynamic batch dimension, dynamic
sequence dimension, and MLM `logits` as the single output.

Excluded (by design): optimizer state, scheduler state, RNG state, the training loss branch,
labels, checkpoint-manager logic, and any Python-dict / variable-length attention-probability
outputs. **The PyTorch checkpoint remains the training source of truth**; the ONNX file is a
derived, inference-only artifact.

## I/O contract

| name | dtype | shape |
|------|-------|-------|
| `input_ids` (in) | int64 | `[batch, sequence]` |
| `attention_mask` (in) | int64 | `[batch, sequence]` |
| `token_type_ids` (in) | int64 | `[batch, sequence]` |
| `logits` (out) | float32 | `[batch, sequence, vocab_size]` |

`batch` and `sequence` are **dynamic** axes. `vocab_size` is fixed (32000 for the 25m config).

## Opset & exporter

**Opset 18.** PyTorch 2.13's ONNX exporter implements opset 18 natively; `onnx>=1.16` and
`onnxruntime>=1.17` support it. Requesting a lower opset triggers a lossy down-conversion, so we
default to 18. In torch 2.x `torch.onnx.export(...)` routes through the dynamo-based exporter
(`torch.export.export`) even when the classic `dynamic_axes` argument is supplied; that path is
what we use and validate here (it prints a note that `dynamic_shapes` is the newer form — the
`dynamic_axes` path still produces a correct, validated graph in this environment). `onnxscript`
is a required dependency of that exporter and is part of the `.[onnx]` extra.

## Install

```bash
python -m pip install -e ".[dev,train,analysis,onnx]"   # adds onnx, onnxruntime, onnxscript
```

The project installs and imports **without** ONNX; ONNX/ORT are imported lazily and only when an
export/inference/validate operation actually runs (missing package → actionable error).

## Export

```bash
python scripts/export_onnx.py \
  --config configs/bert_25m_mac.yaml \
  --checkpoint experiments/smoke/checkpoints \
  --output exports/bert_cord_27m_mlm.onnx \
  --sequence-length 128
```

`--checkpoint` accepts a checkpoint **root** (resolved via `latest.json`), a `step_XXXXXX/`
directory, or a `state.pt` — the same resolution as the rest of the project. Weights are loaded
on CPU. Flags: `--opset`, `--batch-size`, `--static` (disable dynamic axes).

### File size & external data

For the 27.01M model the exporter writes **two files**: a small `.onnx` graph (~0.7 MB) and a
sibling `.onnx.data` weight file (~102 MB). **Both must be shipped together** — ONNX Runtime
loads the weights from `.onnx.data` referenced by the graph. Total artifact ≈ **103 MB** (FP32
weights, same order as the PyTorch checkpoint's model weights). Smaller models keep weights
inline in a single `.onnx`.

## Validate parity (PyTorch vs ONNX)

```bash
python scripts/validate_onnx.py \
  --config configs/bert_25m_mac.yaml \
  --checkpoint experiments/smoke/checkpoints \
  --onnx-model exports/bert_cord_27m_mlm.onnx
```

Runs identical deterministic inputs through both, across ≥2 sequence lengths and ≥2 batch sizes
(including a padded, attention-masked case), and checks: structural validation, shape match,
numerical closeness, exact top-k agreement at masked positions, and no NaN/Inf.

**Tolerances (FP32 CPU): `rtol=1e-3`, `atol=2e-3`.** ONNX export + ONNX Runtime fuse/reorder
float ops and may use different BLAS kernels than eager PyTorch, so bitwise-identical logits are
not expected. On the actual 27.01M model the observed `max|Δ| ≈ 7–8e-6` — three orders of
magnitude below the tolerance — and **top-5 agreement is 1.00**, which is the decisive
correctness signal (softmax/top-k unchanged).

## Run inference (ONNX Runtime)

```bash
python scripts/predict_mask_onnx.py \
  --model exports/bert_cord_27m_mlm.onnx \
  --period 3 --seq-len 24 --topk 5
```

CPU by default. It prints the active and available execution providers, validates input names
and output rank, and prints top-k token ids + probabilities. Explicit ids: `--input "1 5 6 2"`.
No tokenizer dependency — synthetic token ids (matching the current project scope).

### Inspecting available ONNX Runtime providers

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

CPU-only wheels report `['CPUExecutionProvider']` (this environment also shows
`AzureExecutionProvider`).

## Platform expectations

- **CPU (validated here):** `CPUExecutionProvider`, FP32. This is what was actually run and
  measured.
- **Mac (Apple Silicon):** the same CPU path works. A CoreML execution provider exists but has
  **not** been validated here — do not assume CoreML acceleration until run on the Mac.
- **DGX CUDA:** install `onnxruntime-gpu` and select `CUDAExecutionProvider`
  (`--providers CUDAExecutionProvider,CPUExecutionProvider`). This has **not** been run on real
  NVIDIA hardware — `onnxruntime-gpu` is **not** validated. See `docs/DGX_DEPLOYMENT.md`.

## Training checkpoint vs inference artifact

| | PyTorch checkpoint (`step_XXXXXX/`) | ONNX artifact (`exports/*.onnx`) |
|-|-|-|
| authoritative | **yes** (training source of truth) | no (derived) |
| contains | model + optimizer + scheduler + RNG + metadata | inference graph + weights only |
| used for | resume / continue training | portable inference |
| in Git | no (git-ignored) | no (git-ignored) |

## Known limitations

- FP32 CPU only was validated. BF16/FP16 ONNX, CoreML, and CUDA EP are untested here.
- The exporter emits benign warnings (dynamo `dynamic_axes` note, opset messages); the produced
  graph passes `onnx.checker` and ORT execution.
- Task is MLM only. The artifact says nothing about coordination behavior.

## Distribution

`exports/`, `*.onnx`, and `*.onnx.data` are **git-ignored** — do not commit the ~100 MB artifact
to Git history. Distribute release artifacts via **GitHub Releases** or **Hugging Face**, not
ordinary Git. Always ship the `.onnx` graph and its `.onnx.data` weight file together.
