# Current Status

_Last updated: 2026-07-13 — Real-text stage: offline packed-corpus pipeline + loader (unstaged)._

## Real-text pretraining pipeline — built & validated (changes UNSTAGED for review)

- **Test suite: 159 passed, 1 xfailed** (148 + 11 packed-corpus tests; tiny fixtures, offline).
- Rejected article-level truncation. New offline **tokenize-and-pack** pipeline
  (`packed_corpus.py` + `tokenize_and_pack_corpus.py`): `[CLS] content [SEP] PAD` rows in
  `data/tokenized/<run>/{train,validation}/*.npy` (uint16/uint32), long docs split across
  sequences, **no cross-document packing**, atomic shards, manifest with checksums/counters,
  **no stored MLM masks/labels** (masking stays dynamic).
- Memory-mapped `PackedTokenDataset` + `PackedMLMCollator` (reuses `MLMasker`; unchanged MLM
  objective); `DataConfig.packed_dataset_dir`; dispatch priority **packed → HF text → synthetic**.
- `validate_packed_corpus.py` (23/23 on the sample): schema, checksums, dtype/dims, id bounds,
  CLS/SEP/PAD framing, no MASK, no labels, non-empty splits, tokenizer checksum.
- Configs `dgx_real_text_{smoke,pilot,full}.yaml` (separate output dirs, W&B offline; full step
  count is a placeholder, never auto-launched). Verified end-to-end on a packed sample via a tiny
  trainer run (exit 0).
- **No git commit** (per instructions); `data/` + `experiments/` outputs git-ignored; model
  architecture and synthetic checkpoints untouched.

## Immediate next step

On the DGX: freeze the 32k byte-BPE tokenizer, pack the 128-token EN+TH corpus, and run the
smoke → resume → pilot → full acceptance gates (`docs/REAL_TEXT_PRETRAINING.md`). Then Milestone 1.

---

_Prior status (Tokenizer Milestone) below._

## Tokenizer Milestone — earlier
_Last updated: 2026-07-13 — Tokenizer Milestone: reproducible tokenizer pipeline (unstaged)._

## Tokenizer Milestone — pipeline built & validated (changes UNSTAGED for review)

- **Test suite: 148 passed, 1 xfailed** (139 + 9 tokenizer tests; tiny fixtures, offline).
- New: `corpus.py` + `prepare_tokenizer_corpus.py` (read txt/md/jsonl/HF, normalize, dedup,
  deterministic shuffle, language stats, manifest + report); `tokenizer_train.py` + extended
  `train_tokenizer.py` (config-driven byte_bpe / unigram / wordpiece, special tokens pinned to
  ids 0–4); `tokenizer_eval.py` + `evaluate_tokenizer.py`; 3 configs in `configs/tokenizer/`;
  `docs/tokenizer_pipeline.md`, `docs/recommended_corpus.md`.
- Verified on a ~2.6 KB offline sample corpus: all three algorithms train, reserved-token
  integrity OK, byte-BPE 0% UNK / 100% normalized round-trip. Real 32k vocab needs the large
  corpus.
- **Corpus size policy applied:** real multilingual corpora (EN/TH Wikipedia, OSCAR, mC4,
  FineWeb, CC100) all exceed 1 GB → **not downloaded**; DGX download commands in
  `docs/recommended_corpus.md`.
- **No git commit** (per instructions): changes left unstaged; `data/` and `artifacts/` outputs
  are git-ignored. No AutoModel / coordination / language-understanding claim.

## Immediate next step

On the DGX: download a bounded multilingual corpus (`docs/recommended_corpus.md`), train the
three 32k tokenizers, evaluate, and **select + freeze** one for MLM pretraining. Then Milestone 1
(100M).

---

_Prior status (ONNX + HF packaging) below._

## Hugging Face ONNX package (v0.1.2-hf-onnx) — earlier milestone
_Last updated: 2026-07-11 — Milestone 0.7 (ONNX) + Hugging Face ONNX package staged._

## Hugging Face ONNX package (v0.1.2-hf-onnx) — staged locally, not uploaded

- **Test suite: 139 passed, 1 xfailed** (122 + 17 HF-package tests; tiny fixtures, offline).
- Local staging dir `bert-cord-27m-mlm-onnx/` (git-ignored) built via
  `scripts/build_hf_onnx_package.py`; validated **17/17** offline via
  `scripts/validate_hf_onnx_package.py`. Future HF repo `sikkha/bert-cord-27m-mlm-onnx`.
- Contents: README (model card), LICENSE, config.json, evaluation.json, requirements.txt,
  inference.py (standalone, no bert_cord dep), MANIFEST.json (SHA-256), onnx/model.onnx (+ .data).
  Total ≈ 103.2 MB. Packaged graph correctly references `model.onnx.data` (relinked).
- **Separated provenance:** `model_source_commit`/`model_source_tag` (ONNX export commit) are
  tracked distinctly from `packaging_source_commit`/`packaging_source_tag` (tooling commit).
  The builder aborts (non-zero exit) rather than reuse a directory it cannot cleanly remove.
- Parity (packaged ONNX vs PyTorch): max|Δ| 8.11e-6, top-5 agreement 1.00, no NaN/Inf, CPU/FP32.
- **No upload, no HF auth, no remote creation, no network.** Honest model card: synthetic MLM
  baseline, no tokenizer, not a coordinator, no `AutoModel`/language-understanding/production claim.
- Untested (unchanged): CUDA (`onnxruntime-gpu`), CoreML, FP16/BF16.

## Immediate next step

When ready, upload **manually**: `hf repo create sikkha/bert-cord-27m-mlm-onnx --type model` then
`hf upload sikkha/bert-cord-27m-mlm-onnx bert-cord-27m-mlm-onnx .` (see
`docs/HUGGINGFACE_ONNX_RELEASE.md`). Then DGX ONNX-GPU validation; then Milestone 1 (100M).

---

_Milestone 0.7 (ONNX export) status below (still current)._

## Milestone 0.7 (ONNX export & portable inference) — verified

- **Test suite: 122 passed, 1 xfailed** (105 + 17 new ONNX tests; ORT-dependent tests skip if
  packages absent).
- **ONNX export:** actual **27.01M** checkpoint exported to `exports/bert_cord_27m_mlm.onnx`
  (+ `.onnx.data` external weights); total ≈ **103.17 MB**; opset 18; `onnx.checker` PASSED.
- **ONNX Runtime CPU validation:** executes on `CPUExecutionProvider`; dynamic batch+sequence
  confirmed.
- **PyTorch↔ONNX parity:** `max|Δ| ≈ 7–8e-6` (rtol=1e-3, atol=2e-3), **top-5 agreement 1.00**,
  no NaN/Inf, across seq {64,128} × batch {1,3} incl. padded cases. Application-level top-1
  identical between `predict_mask.py` and `predict_mask_onnx.py`.
- Scope: **MLM only.** Optimizer/scheduler/RNG/loss/labels are NOT exported; PyTorch checkpoints
  remain authoritative. Docs: `docs/ONNX_EXPORT.md`.
- **Mac CoreML: untested.** **DGX CUDA (`onnxruntime-gpu` / CUDAExecutionProvider): untested.**
  Only FP32 CPU was actually run/validated.

## Immediate next step

On the DGX: `pip install onnxruntime-gpu`, re-run `scripts/validate_onnx.py` with
`--providers CUDAExecutionProvider,CPUExecutionProvider`, record results (see
`docs/DGX_DEPLOYMENT.md`); optionally test CoreML EP on the Mac. Then Milestone 1 (100M).

---

_Prior release-candidate state below (still current for non-ONNX items)._

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
