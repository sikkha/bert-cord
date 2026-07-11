# bert_cord — compact BERT coordinator (release candidate)

A purpose-built, very small **BERT-style encoder** for AI-coordination research. The long-term
goal (`dev_mem/project_brief.md`) is a compact model (≤ ~200M params) that interprets system
state and decides when to handle an event locally, delegate to a larger LLM, activate memory,
request clarification, or control a task's lifecycle — without doing the full reasoning itself.

**Current stage:** a clean, dual-platform (Apple Silicon **MPS** / NVIDIA **CUDA**) ~25M
(actual **27.01M**) custom BERT **masked-language-model** pretraining system, plus evaluation,
learning-curve analysis, benchmarking, and release tooling. The encoder is implemented from
scratch — **no Hugging Face `BertModel` / `AutoModelForMaskedLM` internals**. HF `datasets`,
`tokenizers`, and `accelerate` are used only for data, tokenization, and the training loop.

> Coordination heads, teacher distillation, external-LLM routing, and voice are **not** part of
> this stage (documented placeholders only).

## What's implemented

- Custom bidirectional multi-head self-attention (explicit softmax; optional PyTorch SDPA).
- Learned token/position/(configurable) token-type embeddings; pre-/post-LN blocks; tied
  input/output embeddings; dynamic 15% MLM with 80/10/10 replacement.
- AdamW + warmup/cosine decay, gradient accumulation, deterministic seeds.
- **Platform-aware runtime**: device (auto→cuda|mps|cpu), BF16-when-supported with fp32
  fallback, TF32/SDPA/pinned-memory/persistent-workers/non-blocking/fused-AdamW/torch.compile —
  all feature-detected, optional, reported at startup, and safely disabled when unavailable.
- **Immutable, checksum-verified checkpoints** (`step_XXXXXX/` + `latest.json` pointer).
- Inference utilities, tiny overfit test, synthetic evaluation, metrics logging.
- **Heuristic learning-curve analysis** with static figures + JSON/Markdown reports.
- **Environment diagnostics** and a **short training benchmark** with a bounded batch probe.
- **ONNX export + portable ONNX Runtime inference** (CPU), with PyTorch↔ONNX parity validation.

## Installation

Core is minimal (`torch`, `numpy`, `pyyaml`). Everything else is an extra.

```bash
# minimal runtime (import + run the model)
python -m pip install -e .

# development + full training + analysis (recommended)
python -m pip install -e ".[dev,train,analysis]"

# optional extras
python -m pip install -e ".[scipy_optional]"   # refines curve fits (numpy path works without)
python -m pip install -e ".[dev,train,analysis,wandb]"   # + optional W&B tracking (never required)
```

Optional experiment tracking is off by default (`tracking.backend: none`). Enable W&B with
`--wandb --wandb-mode offline` (no login/network needed offline); see
[`docs/WANDB_INTEGRATION.md`](docs/WANDB_INTEGRATION.md). Local JSONL metrics + curve analysis
remain the source of truth.

Extras: `dev` (pytest) · `train` (accelerate, datasets, tokenizers, safetensors, psutil) ·
`analysis` (matplotlib) · `scipy_optional` · `wandb` · `all` (= dev+train+analysis).

### MacBook (Apple Silicon) development

```bash
git clone <repo-url> bert_cord && cd bert_cord
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,train,analysis]"
python scripts/check_environment.py            # expect device: mps
python -m pytest -q
python scripts/pretrain_mlm.py --config configs/bert_25m_mac.yaml --smoke --max-steps 40
```

### DGX Spark (CUDA) installation

```bash
git clone <repo-url> bert_cord && cd bert_cord
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,train,analysis]"
python scripts/check_environment.py --require dgx   # must exit 0 on DGX
python -m pytest -q
```

See **`docs/DGX_DEPLOYMENT.md`** for the full DGX procedure, acceptance criteria, and the
conservative edit policy.

## Development pipeline (required order)

1. **environment check** — `scripts/check_environment.py` (`--require dgx` on DGX)
2. **tests** — `python -m pytest -q`
3. **portability benchmark** — `scripts/benchmark_training.py --config configs/bert_25m_dgx_portability.yaml --steps 50 ...`
4. **checkpoint / resume test** — a short smoke run + resume from the checkpoint
5. **throughput benchmark** — `scripts/benchmark_training.py --config configs/bert_25m_dgx_throughput.yaml --steps 200 ...`
6. **only then** longer training.

## Configuration

Scientific settings (`model`, most of `train`) are separate from hardware/runtime (`runtime`,
precision, dataloader perf). Configs compose via an `extends:` list:

```
configs/
├── model/{bert_25m,bert_100m,bert_200m}.yaml        # scientific model definition
├── platform/{mac_mps,dgx_portability,dgx_throughput}.yaml   # runtime + train envelope
├── experiments/{smoke_mac,smoke_dgx,synthetic_500,real_text_sanity}.yaml
├── examples/minimal.yaml
├── bert_25m_mac.yaml            # resolved: model + platform (self-contained)
├── bert_25m_dgx_portability.yaml
├── bert_25m_dgx_throughput.yaml
├── bert_100m_dgx.yaml / bert_200m_dgx.yaml          # PROVISIONAL — do not full-train yet
└── bert_25m.yaml / bert_100m.yaml / bert_200m.yaml  # legacy flat configs (still supported)
```

The **portability** profile keeps the exact same mathematical workload as the Mac run (same
seed/batch/seq/optimizer/scheduler, SDPA off) and differs only in device + precision (bf16).
The **throughput** profile intentionally enables performance features and a larger workload.
The fully resolved settings that materially affect training are **printed at startup**.

## Run

```bash
# Smoke training (synthetic; no network)
python scripts/pretrain_mlm.py --config configs/bert_25m_mac.yaml --smoke --max-steps 40

# Resume (follows latest.json in the checkpoint root, verifies checksum)
python scripts/pretrain_mlm.py --config configs/bert_25m_mac.yaml --smoke --max-steps 80 \
    --resume experiments/smoke_mac/checkpoints

# Inference, synthetic eval, overfit capacity check
python scripts/predict_mask.py --config configs/bert_25m.yaml --checkpoint <ckpt> --topk 5
python scripts/evaluate_synthetic.py --config configs/bert_25m.yaml --checkpoint <ckpt>
python scripts/overfit_tiny.py --config configs/bert_25m.yaml

# Metrics logging + learning-curve analysis
python scripts/pretrain_mlm.py --config configs/bert_25m_mac.yaml --smoke --max-steps 40 \
    --metrics-file experiments/run/metrics.jsonl
python scripts/analyze_training_curve.py --metrics experiments/run/metrics.jsonl \
    --future-step 1000 --future-step 2000 --plot --show-confidence

# Environment report + benchmark
python scripts/check_environment.py --json /tmp/env.json
python scripts/benchmark_training.py --config configs/bert_25m_dgx_portability.yaml \
    --steps 50 --output-dir experiments/benchmarks/portability

# ONNX export + portable inference (needs the `onnx` extra)
python scripts/export_onnx.py --config configs/bert_25m_mac.yaml \
    --checkpoint experiments/smoke/checkpoints --output exports/bert_cord_27m_mlm.onnx
python scripts/validate_onnx.py --config configs/bert_25m_mac.yaml \
    --checkpoint experiments/smoke/checkpoints --onnx-model exports/bert_cord_27m_mlm.onnx
python scripts/predict_mask_onnx.py --model exports/bert_cord_27m_mlm.onnx --period 3 --seq-len 24
```

ONNX export/inference is optional (`pip install -e ".[onnx]"`) and MLM-only; the PyTorch
checkpoint stays authoritative. See [`docs/ONNX_EXPORT.md`](docs/ONNX_EXPORT.md).

## Experiment output layout

```
experiments/<run_id>/
├── environment.json          resolved_config.yaml   metrics.jsonl
├── analysis/{analysis_summary.json, training_curve_report.md, plots/}
├── checkpoints/{step_XXXXXX/, latest.json}
└── run_report.md
```

Generated experiment outputs are **git-ignored**; a tiny fixture
(`tests/fixtures/curve_metrics.jsonl`) is tracked for testing the analyzer.

## Status

Verified on CPU/fp32 (development) and Apple-Silicon-ready. **88 tests pass (+1 intentional
xfail).** The CUDA/BF16 path is feature-detected and unit-tested via mocks but has **not** been
run on real NVIDIA hardware yet — that is **DGX readiness, not DGX validation** (see
`docs/DGX_DEPLOYMENT.md`). Docs on the analyzer: `docs/training_curve_analysis.md`; release
steps: `docs/RELEASE_CHECKLIST.md`.

## License

Apache-2.0. Reference implementations consulted (ideas only, not copied): `barneyhill/minBERT`,
Hugging Face Transformers `run_mlm_no_trainer.py`, `google-research/bert`.
