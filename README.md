# BERT-Cord — Compact BERT Coordinator Research Framework

[![Release](https://img.shields.io/github/v/release/sikkha/bert-cord?display_name=tag)](https://github.com/sikkha/bert-cord/releases)
[![License](https://img.shields.io/github/license/sikkha/bert-cord)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2018-005CED.svg?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-27M%20ONNX%20Model-yellow)](https://huggingface.co/sikkha/bert-cord-27m-mlm-onnx)

A purpose-built, compact **BERT-style encoder** for AI-coordination research.

The long-term goal, documented in [`dev_mem/project_brief.md`](dev_mem/project_brief.md), is a model of no more than approximately 200 million parameters that interprets system state and helps determine whether an event should be:

- handled locally;
- delegated to a larger language model;
- connected to semantic or working memory;
- clarified with the user;
- routed to a tool;
- or used to update the lifecycle of an ongoing task.

The coordinator is not intended to perform all reasoning itself. It is intended to become a small learned control component inside a larger modular cognitive architecture.

> **Current scope:** the released 27.01M model is a synthetic masked-language-model baseline. It is not yet the mini-amygdala coordinator and currently performs no learned delegation, routing, memory activation, task control, or general-language understanding.

## Public resources

- **Source repository:** [github.com/sikkha/bert-cord](https://github.com/sikkha/bert-cord)
- **Latest GitHub release:** [BERT-Cord 27M ONNX Baseline](https://github.com/sikkha/bert-cord/releases)
- **Hugging Face model:** [sikkha/bert-cord-27m-mlm-onnx](https://huggingface.co/sikkha/bert-cord-27m-mlm-onnx)
- **Experiment tracking:** [Weights & Biases — bert-cord](https://wandb.ai/sikkha/bert-cord)

## Current stage

The project currently provides a dual-platform, approximately 25M-parameter custom BERT implementation with an actual parameter count of:

```text
27,010,304 parameters
```

The current system includes:

- masked-language-model pretraining;
- deterministic synthetic experiments;
- checkpoint and resume support;
- inference and evaluation utilities;
- learning-curve analysis;
- W&B integration;
- platform-aware Mac and DGX configuration;
- ONNX export;
- ONNX Runtime inference;
- reproducible Hugging Face packaging.

The encoder is implemented from scratch. It does **not** use Hugging Face `BertModel` or `AutoModelForMaskedLM` internals.

Hugging Face `datasets`, `tokenizers`, and `accelerate` are used only for data preparation, tokenization, and training infrastructure.

Teacher distillation, coordination heads, external-LLM routing, persistent memory activation, and voice integration remain future milestones.

## Milestones

- ✅ Milestone 0 — custom 27M BERT MLM foundation
- ✅ Platform-aware Mac MPS and DGX CUDA runtime
- ✅ Checkpoint, resume, and deterministic evaluation
- ✅ Learning-curve analysis and training diagnostics
- ✅ Weights & Biases integration
- ✅ ONNX export and portable ONNX Runtime inference
- ✅ Reproducible Hugging Face ONNX packaging
- 🚧 DGX validation and larger-scale training
- ⏳ Teacher distillation
- ⏳ Learned coordination heads
- ⏳ External-LLM and tool routing
- ⏳ Mini-amygdala coordinator
- ⏳ Integration into AI Blue

```text
27M MLM baseline
       │
       ▼
DGX validation and scaling
       │
       ▼
Teacher distillation
       │
       ▼
Coordination heads
       │
       ▼
Mini-amygdala coordinator
       │
       ▼
AI Blue cognitive architecture
```

## What is implemented

- Custom bidirectional multi-head self-attention.
- Explicit attention softmax with optional PyTorch SDPA.
- Learned token, position, and configurable token-type embeddings.
- Configurable pre-layer-normalization and post-layer-normalization blocks.
- Tied input and output embeddings.
- Dynamic 15% masked-language-model corruption using the standard 80/10/10 replacement policy.
- AdamW optimization with warm-up and cosine decay.
- Gradient accumulation and deterministic random seeds.
- Platform-aware runtime resolution:
  - CUDA;
  - Apple MPS;
  - CPU fallback.
- Feature-detected and optional:
  - BF16;
  - TF32;
  - SDPA;
  - pinned memory;
  - persistent workers;
  - non-blocking transfer;
  - fused AdamW;
  - `torch.compile`.
- Immutable, checksum-verified checkpoints using:
  - `step_XXXXXX/`;
  - `latest.json`.
- Synthetic inference and evaluation utilities.
- Tiny-batch overfit capacity test.
- Metrics logging and static learning-curve reports.
- Environment diagnostics and bounded batch probing.
- Optional W&B online and offline experiment tracking.
- ONNX opset-18 export with dynamic batch and sequence axes.
- PyTorch-to-ONNX numerical and top-k parity validation.
- Standalone ONNX Runtime inference.
- Reproducible Hugging Face model-repository packaging.

## Installation

The core installation is intentionally minimal:

- `torch`
- `numpy`
- `pyyaml`

Everything else is provided through optional extras.

```bash
# Minimal runtime
python -m pip install -e .

# Development, tests, training, and analysis
python -m pip install -e ".[dev,train,analysis]"

# ONNX export and ONNX Runtime inference
python -m pip install -e ".[onnx]"

# Optional W&B tracking
python -m pip install -e ".[dev,train,analysis,wandb]"

# Optional SciPy refinement for learning-curve fitting
python -m pip install -e ".[scipy_optional]"

# Full development, training, analysis, and ONNX environment
python -m pip install -e ".[all]"
```

Available extras:

| Extra | Purpose |
|---|---|
| `dev` | Pytest and development checks |
| `train` | Accelerate, datasets, tokenizers, safetensors, and system reporting |
| `analysis` | Matplotlib learning-curve visualization |
| `onnx` | ONNX export, ONNX Runtime inference, and ONNX Script |
| `wandb` | Optional Weights & Biases experiment tracking |
| `scipy_optional` | Optional refinement of curve fitting |
| `all` | `dev + train + analysis + onnx` |

W&B is disabled by default:

```yaml
tracking:
  backend: none
```

Enable offline tracking without login or network access:

```bash
python scripts/pretrain_mlm.py \
  --config configs/bert_25m_mac.yaml \
  --smoke \
  --max-steps 40 \
  --wandb \
  --wandb-mode offline
```

See [`docs/WANDB_INTEGRATION.md`](docs/WANDB_INTEGRATION.md).

Local JSONL metrics and generated reports remain the authoritative local experiment records.

## MacBook development

```bash
git clone git@github.com:sikkha/bert-cord.git
cd bert-cord

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,train,analysis,onnx]"

python scripts/check_environment.py
python -m pytest -q

python scripts/pretrain_mlm.py \
  --config configs/bert_25m_mac.yaml \
  --smoke \
  --max-steps 40
```

The expected resolved device on Apple Silicon is:

```text
mps
```

## DGX Spark installation

```bash
git clone git@github.com:sikkha/bert-cord.git
cd bert-cord

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,train,analysis,wandb,onnx]"

python scripts/check_environment.py --require dgx
python -m pytest -q
```

See [`docs/DGX_DEPLOYMENT.md`](docs/DGX_DEPLOYMENT.md) for the complete deployment procedure, acceptance criteria, and conservative DGX edit policy.

## Required development pipeline

Run validation in this order:

1. **Environment check**

   ```bash
   python scripts/check_environment.py --require dgx
   ```

2. **Test suite**

   ```bash
   python -m pytest -q
   ```

3. **Portability benchmark**

   ```bash
   python scripts/benchmark_training.py \
     --config configs/bert_25m_dgx_portability.yaml \
     --steps 50 \
     --output-dir experiments/benchmarks/dgx_portability
   ```

4. **Checkpoint and resume test**

5. **Throughput benchmark**

   ```bash
   python scripts/benchmark_training.py \
     --config configs/bert_25m_dgx_throughput.yaml \
     --steps 200 \
     --output-dir experiments/benchmarks/dgx_throughput
   ```

6. **Longer training only after all earlier gates pass**

## Configuration

Scientific settings are separated from runtime and hardware settings.

Configs compose using `extends:`:

```text
configs/
├── model/
│   ├── bert_25m.yaml
│   ├── bert_100m.yaml
│   └── bert_200m.yaml
├── platform/
│   ├── mac_mps.yaml
│   ├── dgx_portability.yaml
│   └── dgx_throughput.yaml
├── experiments/
│   ├── smoke_mac.yaml
│   ├── smoke_dgx.yaml
│   ├── synthetic_500.yaml
│   └── real_text_sanity.yaml
├── examples/
│   └── minimal.yaml
├── bert_25m_mac.yaml
├── bert_25m_dgx_portability.yaml
├── bert_25m_dgx_throughput.yaml
├── bert_100m_dgx.yaml
├── bert_200m_dgx.yaml
├── bert_25m.yaml
├── bert_100m.yaml
└── bert_200m.yaml
```

The 100M and 200M DGX profiles remain provisional.

The portability profile preserves the same mathematical workload as the Mac run while changing only device and precision settings.

The throughput profile intentionally enables performance features and a larger workload.

Resolved settings that materially affect training are printed at startup.

## Training and evaluation

### Smoke training

```bash
python scripts/pretrain_mlm.py \
  --config configs/bert_25m_mac.yaml \
  --smoke \
  --max-steps 40
```

### Resume training

```bash
python scripts/pretrain_mlm.py \
  --config configs/bert_25m_mac.yaml \
  --smoke \
  --max-steps 80 \
  --resume experiments/smoke_mac/checkpoints
```

Checkpoint roots are resolved through `latest.json`, with checksum verification.

### PyTorch inference

```bash
python scripts/predict_mask.py \
  --config configs/bert_25m.yaml \
  --checkpoint <checkpoint-path> \
  --topk 5
```

### Synthetic evaluation

```bash
python scripts/evaluate_synthetic.py \
  --config configs/bert_25m.yaml \
  --checkpoint <checkpoint-path>
```

### Tiny overfit-capacity test

```bash
python scripts/overfit_tiny.py \
  --config configs/bert_25m.yaml
```

## Metrics and learning-curve analysis

Generate metrics:

```bash
python scripts/pretrain_mlm.py \
  --config configs/bert_25m_mac.yaml \
  --smoke \
  --max-steps 40 \
  --metrics-file experiments/run/metrics.jsonl
```

Analyze and visualize:

```bash
python scripts/analyze_training_curve.py \
  --metrics experiments/run/metrics.jsonl \
  --future-step 1000 \
  --future-step 2000 \
  --plot \
  --show-confidence
```

The forecasting component is heuristic. It does not guarantee the optimal stopping point and should not replace scientific validation.

See [`docs/training_curve_analysis.md`](docs/training_curve_analysis.md).

## Environment diagnostics and benchmarking

```bash
python scripts/check_environment.py \
  --json /tmp/bert-cord-environment.json
```

```bash
python scripts/benchmark_training.py \
  --config configs/bert_25m_dgx_portability.yaml \
  --steps 50 \
  --output-dir experiments/benchmarks/portability
```

## ONNX export and portable inference

Install the ONNX extra:

```bash
python -m pip install -e ".[onnx]"
```

Export:

```bash
python scripts/export_onnx.py \
  --config configs/bert_25m_mac.yaml \
  --checkpoint experiments/smoke/checkpoints \
  --output exports/bert_cord_27m_mlm.onnx
```

Validate PyTorch-to-ONNX parity:

```bash
python scripts/validate_onnx.py \
  --config configs/bert_25m_mac.yaml \
  --checkpoint experiments/smoke/checkpoints \
  --onnx-model exports/bert_cord_27m_mlm.onnx
```

Run ONNX Runtime inference:

```bash
python scripts/predict_mask_onnx.py \
  --model exports/bert_cord_27m_mlm.onnx \
  --period 3 \
  --seq-len 24 \
  --topk 5
```

The exported model uses:

```text
ONNX opset:       18
Precision:        FP32
Dynamic batch:    yes
Dynamic sequence: yes
Validated runtime: ONNX Runtime CPUExecutionProvider
```

The 27M export consists of two files:

```text
bert_cord_27m_mlm.onnx
bert_cord_27m_mlm.onnx.data
```

Both files are required.

The PyTorch checkpoint remains the authoritative training source. ONNX is a derived inference artifact.

See [`docs/ONNX_EXPORT.md`](docs/ONNX_EXPORT.md).

## Hugging Face ONNX package

The released ONNX package is available at:

[https://huggingface.co/sikkha/bert-cord-27m-mlm-onnx](https://huggingface.co/sikkha/bert-cord-27m-mlm-onnx)

Build a fresh local package:

```bash
python scripts/build_hf_onnx_package.py \
  --config configs/bert_25m_mac.yaml \
  --onnx-model exports/bert_cord_27m_mlm.onnx \
  --output dist/bert-cord-27m-mlm-onnx \
  --repo-id sikkha/bert-cord-27m-mlm-onnx \
  --package-version 0.1.2-hf-onnx \
  --model-source-commit 0e17db558ebcce29f40b49d546af8b2704640230 \
  --model-source-tag v0.1.1-onnx
```

Validate the package:

```bash
python scripts/validate_hf_onnx_package.py \
  dist/bert-cord-27m-mlm-onnx
```

Run standalone packaged inference:

```bash
python dist/bert-cord-27m-mlm-onnx/inference.py
```

The package records separate provenance for:

- the model-export source;
- the packaging-tool source;
- the model source tag;
- the packaging release tag.

This is a generic ONNX Runtime model repository. It is not compatible with `transformers.AutoModel` and includes no natural-language tokenizer.

See [`docs/HUGGINGFACE_ONNX_RELEASE.md`](docs/HUGGINGFACE_ONNX_RELEASE.md).

## Experiment output layout

```text
experiments/<run_id>/
├── environment.json
├── resolved_config.yaml
├── metrics.jsonl
├── analysis/
│   ├── analysis_summary.json
│   ├── training_curve_report.md
│   └── plots/
├── checkpoints/
│   ├── step_XXXXXX/
│   └── latest.json
└── run_report.md
```

Generated experiment outputs are excluded from Git.

A small fixture at:

```text
tests/fixtures/curve_metrics.jsonl
```

is tracked for learning-curve analyzer tests.

## Verified status

Current verified test result:

```text
139 passed, 1 intentional xfail
```

Validated:

- Apple Silicon MPS training path;
- CPU FP32 development;
- checkpoint and resume behavior;
- synthetic evaluation;
- W&B online and offline tracking;
- ONNX structural validation;
- ONNX Runtime CPU inference;
- dynamic batch and sequence axes;
- PyTorch-to-ONNX top-5 parity;
- reproducible Hugging Face packaging;
- package checksum and leak validation;
- standalone downloaded-model inference path.

Not yet validated on physical DGX hardware:

- real CUDA/BF16 training;
- ONNX Runtime CUDAExecutionProvider;
- FP16 ONNX inference;
- BF16 ONNX inference;
- TensorRT;
- CoreML execution provider.

The CUDA and BF16 paths are feature-detected and covered through configuration and mocked tests, but this is currently **DGX readiness**, not yet **DGX validation**.

## Release history

| Release | Description |
|---|---|
| `v0.1.0-dgx-ready` | Mac-verified baseline prepared for DGX deployment |
| `v0.1.1-onnx` | ONNX export and portable inference |
| `v0.1.2-hf-package` | Reproducible Hugging Face ONNX package tooling |

## License

Apache-2.0.

Reference implementations consulted for architectural ideas only; no source was copied:

- `barneyhill/minBERT`
- Hugging Face Transformers `run_mlm_no_trainer.py`
- `google-research/bert`
