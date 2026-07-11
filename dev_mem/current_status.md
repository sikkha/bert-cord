# Current Status

_Last updated: 2026-07-11 — Pre-GitHub release candidate (dual-platform Mac/DGX; DGX-ready)._

## Release-prep state (verified)

- **88 passed, 1 xfailed.** Actual parameter count **27,010,304 (~27.01M)**.
- Packaging: minimal core + extras `dev/train/analysis/scipy_optional/wandb/all`;
  `pip install -e ".[dev,train,analysis]"` verified.
- Platform-aware runtime (`runtime.py`): device auto→cuda|mps|cpu, BF16-when-supported→fp32
  fallback, TF32/SDPA/pinned/persistent/non-blocking/fused-AdamW/torch.compile feature-detected,
  optional, reported at startup, safely disabled. torch.compile off by default; no FlashAttn dep.
- Config: `model|platform|experiments|examples` + resolved `bert_25m_{mac,dgx_portability,
  dgx_throughput}` (+100m/200m dgx) via `extends`; portability == Mac math except precision.
- Immutable checksum-verified checkpoints (`step_XXXXXX/` + `latest.json`, atomic, SHA-256,
  verify-on-load); resume follows latest.json.
- Diagnostics (`check_environment.py` text+JSON, `--require training|dgx`) and benchmark
  (`benchmark_training.py`, ≤200 steps, OOM-safe bounded batch probe) with full output bundle.
- Docs: README dual-platform pipeline, `docs/DGX_DEPLOYMENT.md`, `docs/RELEASE_CHECKLIST.md`,
  conservative DGX edit policy in `CLAUDE.md`.

## Not done here (honest)

- **CUDA/BF16/MPS not executed on real hardware** — readiness only, validate on DGX per
  `docs/DGX_DEPLOYMENT.md`. No 100M/200M full training. Coordination/distillation/routing/voice
  remain placeholders.
- Git commit/tag pending on the real Mac (this synced sandbox mount blocks commits via a stale
  `.git/HEAD.lock`).

## Immediate next task

On the Mac: `rm -f .git/HEAD.lock` (if no git running) → `git add -A` →
`git rm --cached experiments/smoke/smoke_train.log` → commit → tag `v0.1.0-rc1` → push. Then run
the DGX bring-up sequence.

---

_History below (unchanged)._

## Prior status: Milestone 0 + 0.5 + 0.6

## Milestone 0.6 (training-curve analysis + visualization) — verified

- `src/coordinator_bert/curve_analysis.py` (analysis → structured data) +
  `src/coordinator_bert/curve_plots.py` (matplotlib Agg, headless) +
  `scripts/analyze_training_curve.py` (CLI) + `docs/training_curve_analysis.md`.
  Trainer gained `--metrics-file` and `--early-stop-policy off|warn|stop` (default **off**).
  No model-architecture files changed.
- Test suite now **65 passed, 1 xfailed** (30 new: analysis status logic + robustness, headless
  plotting, summary/report). scipy absent → the numpy grid+lstsq fit path is what runs.
- Conservative by design: heuristic extrapolation, no ML forecaster, no optimality claim;
  forecasts suppressed under detected instability; `stop` policy is guarded and always
  checkpoints first. Status ∈ {CONTINUE, PLATEAU, UNSTABLE, INSUFFICIENT_DATA}.
- Example artifacts under `experiments/run_001/analysis/` (6 figures + summary + report),
  validation-loss figure visually inspected.

---

_Milestone 0.5 status below (unchanged)._

## Milestone 0.5 (evaluation utilities) — verified

- `src/coordinator_bert/inference.py` + `scripts/{predict_mask,overfit_tiny,evaluate_synthetic}.py`
  + `tests/test_inference.py`. No model-architecture files changed.
- Test suite now **35 passed, 1 xfailed** (9 new inference tests).
- `overfit_tiny.py`: **PASS**, masked top-1 = 1.000, loss ≈ 0.0002 — overfit *capacity* proven
  (needs warmup + grad-clip, now defaults).
- `evaluate_synthetic.py`: random-init top-1 = 0.000 (zero-point); ~40-step checkpoint top-1
  ≈ 0.027 / top-5 ≈ 0.055 — measurement discriminates trained vs random.
- `predict_mask.py`: top-k masked-prediction mechanics verified end-to-end.
- Scope respected: **no language-understanding claim**; validates inference, capacity, and
  synthetic-generalization *measurement* only.
- Caveat: strong synthetic generalization not reached within the CPU/45s budget; a persisted
  well-trained checkpoint could not be produced due to a synced-mount overwrite-persistence
  quirk (single-process results unaffected). Both deferred to GPU (DGX Spark).

---

_Milestone 0 status below (unchanged)._

## What currently works (verified by inspected output)

- Custom BERT MLM stack (from scratch, no HF `BertModel`): embeddings, bidirectional
  multi-head self-attention, pre/post-LN transformer blocks, encoder, pooler, MLM head with
  tied input/output embeddings.
- Actual parameter count: **27,010,304 (~27.01M)** — closed-form estimate matches direct count.
- Dynamic MLM masking: ~15% selection, 80/10/10 replacement, special tokens never masked,
  −100 labels for unselected — statistically verified.
- Training loop with Accelerate: AdamW (decoupled weight decay), warmup + cosine/linear decay,
  gradient accumulation, deterministic seeds, startup environment report, val loss +
  masked-token accuracy.
- BF16-when-available with safe fp32 fallback (resolved from real hardware support).
- Checkpoint save + resume: model / optimizer / scheduler / global step / RNG. Resume verified
  to continue without loss spike; determinism test passes on CPU.
- Full test suite: **26 passed, 1 xfailed** (the intentional distillation placeholder).
- Smoke training + resume: loss 10.41 → 6.11, checkpoint at `experiments/smoke/checkpoints/last`.

## What is partially working / limited

- **Masked-token accuracy is low/coarse (~0–2%)** in the smoke run: the eval set has only ~100
  masked tokens and top-1 over the 32000-vocab is hard after ~40 tiny steps. Loss/perplexity
  are the reliable signals; accuracy would improve with more steps / a larger eval set.
- **CUDA BF16 path is untested on real hardware** — this environment is CPU-only aarch64.
  Guarded and fp32-fallback verified, but the DGX Spark bf16 run remains to be done.

## What is broken

- Nothing known. All implemented features pass their tests and the smoke run.

## Placeholders (intentionally NOT implemented in Milestone 0)

- `distillation.py`, `coordination_heads.py`, `scripts/distill_teacher.py`,
  `scripts/train_coordinator.py`, `tests/test_distillation.py` — guarded placeholders that
  raise `NotImplementedError` / exit with a message. Milestones 2–3.

## Latest verified commands

```
python3 -m pytest -q                                   # 26 passed, 1 xfailed
python3 scripts/pretrain_mlm.py --config configs/bert_25m.yaml --smoke --max-steps 20
python3 scripts/pretrain_mlm.py --config configs/bert_25m.yaml --smoke --max-steps 40 \
    --resume experiments/smoke/checkpoints/last
```

## Latest verified checkpoint

`experiments/smoke/checkpoints/last` — `global_step=40`, precision fp32, seed 42.

## Immediate next task

Milestone 1: scale the verified architecture/loop to the ~100M config (`configs/bert_100m.yaml`),
optionally wire a real HF text corpus + trained tokenizer via the existing
`build_text_dataloaders` path, and validate the CUDA BF16 path on the DGX Spark.
