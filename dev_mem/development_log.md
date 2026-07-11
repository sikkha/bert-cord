# Development Log

Append-only chronological record. Never overwrite history.

---

## 2026-07-11 — Session start: Milestone 0 initialization

### Environment inspection (verified)

- OS / arch: Linux 6.8.0-124-generic, **aarch64** (Ubuntu 22.04.5 LTS)
- Python: 3.10.12
- PyTorch: 2.13.0+cpu
- CUDA version (torch.version.cuda): None
- CUDA available (torch.cuda.is_available()): **False** (no GPU in this build/sandbox)
- CUDA device: none
- BF16 support: no CUDA BF16 here; CPU can cast to bfloat16 but training will use fp32 fallback
- Disk space (workspace mount): ~14 GB available
- Git state: initialized fresh repo (no prior commits)
- Libraries: datasets 5.0.0, tokenizers 0.23.1, accelerate 1.14.0, pyyaml, pytest 9.1.1

**Note:** Intended production hardware is an NVIDIA DGX Spark (CUDA + BF16). The code detects
CUDA/BF16 at runtime and falls back to fp32 on CPU. Smoke training in this session therefore
runs on CPU/fp32; the BF16 path is guarded by `torch.cuda.is_bf16_supported()`.

### Implementation plan (Milestone 0)

Goal: a clean, runnable, configurable ~25M-parameter custom BERT MLM pretraining system.
No Hugging Face `BertModel` / `AutoModelForMaskedLM` internals. HF datasets/tokenizers/
Accelerate permitted.

1. **Scaffold** the full repository tree (src package, scripts, tests, configs, dev_mem,
   experiments/smoke), `pyproject.toml`, `CLAUDE.md`, `README.md`, `.gitignore`.
2. **configuration.py** — frozen dataclasses `ModelConfig`, `TrainConfig`, `RunConfig`; YAML
   load/validate; parameter-count estimator.
3. **embeddings.py** — learned token + position + (configurable) token-type embeddings,
   LayerNorm, dropout.
4. **attention.py** — bidirectional multi-head self-attention with additive attention mask;
   optional SDPA; separate output projection + residual/LayerNorm handled in the block.
5. **model.py** — pre-LN or post-LN transformer blocks (config-driven), encoder stack,
   pooler, `BertModel` (custom) + `BertForMaskedLM` wrapper, deterministic init,
   `count_parameters()`.
6. **mlm_head.py** — transform dense + GELU + LayerNorm + decoder tied to token embeddings +
   independent output bias.
7. **masking.py** — dynamic 15% selection, 80/10/10 replacement, never mask special tokens,
   labels −100 for unselected.
8. **data.py** — synthetic dataset + optional HF-dataset text loader, collator using masking.
9. **checkpointing.py** — save/load model, optimizer, scheduler, scaler, global step, RNG
   state, config, metadata.
10. **distillation.py / coordination_heads.py** — documented placeholders (raise
    NotImplementedError). No teacher logic in Milestone 0.
11. **scripts**: `train_tokenizer.py` (HF tokenizers), `pretrain_mlm.py` (Accelerate loop:
    AdamW, warmup+cosine decay, grad accumulation, BF16-when-available, eval loss + masked
    accuracy, checkpoint save/resume, seeding, startup report), `evaluate.py`. Placeholders
    for `distill_teacher.py`, `train_coordinator.py`.
12. **tests**: attention shapes/mask/bidirectionality/NaN/determinism; masking stats/special
    tokens/−100; model shapes/tied weights/grad/param-count; checkpoint save-reload/resume.
    Placeholder `test_distillation.py` (skipped).
13. **Verify**: run full pytest; run smoke training; resume from checkpoint; inspect real
    outputs; record everything in dev_mem.

### Proposed 25M configuration

vocab 32000, hidden 384, 8 layers, 6 heads, intermediate 1536, max_pos 512, type_vocab 2,
post-LN (baseline). Estimated ~26.9M params (tied embeddings). Actual count computed and
recorded in architecture_decisions.md (ADR-002).

### Identified risks

- **Param target drift**: token embedding (vocab×hidden = 12.3M) dominates; "25M" is
  approximate. Mitigation: compute + document actual count, keep within 20–30M micro band.
- **No CUDA/BF16 here**: cannot exercise the GPU BF16 path in this session. Mitigation:
  guard with runtime checks, fall back to fp32, document that BF16 remains untested on GPU.
- **Post-LN stability**: deep post-LN can be unstable; smoke run is short. Mitigation:
  offer pre-LN via config; keep LR/warmup conservative in smoke config.
- **SDPA vs manual attention parity**: must match. Mitigation: default to manual path,
  unit-test both produce finite, correctly-shaped, mask-respecting outputs.
- **Determinism**: full CUDA determinism not guaranteed. Mitigation: seed all RNGs, save RNG
  state in checkpoints, assert resume continuity on step counter & param values.

---

## 2026-07-11 — Milestone 0 implementation & verification

Task attempted: implement + verify the full Milestone 0 custom BERT MLM system.

Files created:
- `pyproject.toml`, `CLAUDE.md`, `README.md`, `.gitignore`
- `configs/bert_25m.yaml` (real), `configs/bert_100m.yaml`, `configs/bert_200m.yaml`
  (provisional, not trained)
- `src/coordinator_bert/`: `__init__.py`, `configuration.py`, `embeddings.py`, `attention.py`,
  `model.py`, `mlm_head.py`, `masking.py`, `data.py`, `checkpointing.py`,
  `distillation.py` (placeholder), `coordination_heads.py` (placeholder)
- `scripts/`: `pretrain_mlm.py`, `train_tokenizer.py`, `evaluate.py`,
  `distill_teacher.py` (placeholder), `train_coordinator.py` (placeholder)
- `tests/`: `conftest.py`, `test_attention.py`, `test_masking.py`, `test_model_shapes.py`,
  `test_checkpoint.py`, `test_distillation.py` (placeholder/xfail)

Commands run (all inspected):
- `pip install torch(cpu) datasets tokenizers accelerate pyyaml pytest` — OK.
- `git init` + commit `52fe1c0`.
- Param check: estimate == count == **27,010,304 (~27.01M)**; tied embeddings confirmed
  (shared storage); random-init forward loss ≈ 10.46 ≈ ln(32000); no NaNs.
- `python3 -m pytest -q` → **26 passed, 1 xfailed** (distillation placeholder xfail).
- Smoke stage A (`--smoke --max-steps 20`) → exit 0, val loss 9.68→7.65, checkpoint at step 20.
- Smoke stage B (`--smoke --max-steps 40 --resume .../last`) → exit 0, restored step 20,
  val loss →6.11 at step 40, no resume spike.

Results: all tests pass; smoke training shows monotonically decreasing loss and a verified
resume. See `experiment_log.md` (EXP-001) for full inspected output and metrics.

Failures / iteration notes (recorded honestly):
- First `test_gradient_propagation` failed: pooler params received no gradient. Correct
  behavior — the pooler is not on the MLM loss path. Refined the test to exclude the pooler
  and added `test_pooler_not_in_mlm_loss_path` to document it.
- Several early smoke runs hit the 45s shell wall (exit 124/137): caused by (a) transient OOM
  from overlapping timed-out processes on a 3.9 GB no-swap box, and (b) periodic checkpoints
  double-writing ~323 MB each on a slow, 98%-full disk. Fixed by disabling periodic saves in
  the smoke profile (single final `last` write) and staging train/resume as two calls.
- Two stale checkpoint dirs (`step_5`, `step_20`) from those timed-out runs cannot be
  `unlink`ed on this mount ("Operation not permitted"); they are git-ignored and harmless.
- Synthetic corpus changed from pure-random to a learnable copy-motif (ADR-008) so the smoke
  run demonstrates real optimization rather than a flat loss.

Environment note: CPU-only aarch64, fp32. The **CUDA BF16 path is guarded and fp32-fallback
verified but not exercised on real GPU hardware** — deferred to the DGX Spark.

Unresolved questions / next step: validate the BF16 path on the DGX Spark; optionally wire a
real HF corpus + trained tokenizer; then Milestone 1 (scale to ~100M using the same loop).

---

## 2026-07-11 — Milestone 0.5: evaluation utilities (no architecture change)

Task attempted: add inference/evaluation utilities without touching the model architecture.

Files created:
- `src/coordinator_bert/inference.py` — helpers: `load_model_for_inference`,
  `find_masked_positions`, `apply_mask_at`, `topk_predictions`, `predict_masked_topk`,
  `masked_accuracy_topk`. (Pure inference/post-processing; no architecture change.)
- `scripts/predict_mask.py` — load checkpoint, accept explicit token ids or a synthetic motif
  sequence, mask position(s), print top-k predictions + probabilities.
- `scripts/overfit_tiny.py` — fixed tiny synthetic batch, train to overfit (warmup + grad-clip),
  report loss + masked top-1/top-5, **exit non-zero if top-1 ≤ threshold**.
- `scripts/evaluate_synthetic.py` — evaluate on unseen synthetic motif sequences across
  multiple periods and sequence lengths; report loss, top-1, top-5 per combo + overall.
- `tests/test_inference.py` — checkpoint loading restores weights, eval-mode determinism,
  top-k output shape, masked-position extraction (+ empty-mask and top-k-accuracy cases).
Files modified (no architecture change): `scripts/pretrain_mlm.py` (added `--eval-every`
override and single-write `_save` already in M0), `scripts/overfit_tiny.py` (added warmup +
grad-clip after diagnosing post-LN instability).

Commands run (all outputs inspected):
- `python3 -m pytest` → **35 passed, 1 xfailed** (9 new inference tests).
- `overfit_tiny.py` (defaults) → **PASS**, masked top-1 = 1.0000, loss 0.0002, 16.9s.
- `evaluate_synthetic.py` random-init → top-1 0.0000 / loss ≈ 10.45 (metric zero-point).
- `evaluate_synthetic.py` on ~40-step smoke checkpoint → top-1 ≈ 0.027, top-5 ≈ 0.055,
  loss ≈ 9.80 (above baseline, honestly weak — undertrained checkpoint).
- `predict_mask.py` on smoke checkpoint → correct top-k mechanics; predictions near-uniform
  (checkpoint barely trained). See experiment_log.md EXP-002 for full inspected output.

Findings / honesty notes:
- Post-LN overfit was unstable at lr 2e-3 with no warmup (plateaued top-1 ≈ 0.15); a 20-step
  linear warmup + gradient clipping fixed it to top-1 = 1.0. Now the script defaults.
- Overfit *capacity* is proven (top-1 = 1.0); strong synthetic *generalization* needs more
  training steps than the CPU/45s-per-call budget allows — a fresh 60-step attempt plateaued
  at loss ≈ ln(64) (learned the sub-vocab, not yet the exact copy). Deferred to GPU.
- **Checkpoint persistence caveat:** this synced mount reliably creates new files but does not
  reliably persist *overwrites* of the 323 MB `state.pt` across separate runs (its meta was
  seen at step 4 → 40 → 999). This never affected single-process results (each script loads +
  uses a checkpoint within one process) but blocked producing a persisted well-trained
  checkpoint here. Non-issue on a normal filesystem / the DGX Spark.

No architecture files under `src/coordinator_bert/{model,attention,embeddings,mlm_head}.py`
were changed. Milestone 0.5 goals (validate inference, overfit capacity, synthetic-generalization
measurement) are met.

Next step: unchanged — validate BF16 on the DGX Spark and begin Milestone 1 (100M).

---

## 2026-07-11 — Milestone 0.6: training-curve analysis + visualization (no architecture change)

Task attempted: add a conservative, non-learned training-curve analysis utility with static
figures, plus optional opt-in early-stop integration.

Files created:
- `src/coordinator_bert/curve_analysis.py` — load CSV/JSONL, EMA, step/log-step slopes,
  plateau + instability detection, three closed-form curve fits (power / exp / inverse-sqrt)
  via numpy grid+lstsq (scipy optional), tail-RMSE/AIC selection, bootstrap CIs, status
  {CONTINUE, PLATEAU, UNSTABLE, INSUFFICIENT_DATA}. Returns structured data only.
- `src/coordinator_bert/curve_plots.py` — matplotlib `Agg` (headless); six separate figures
  (validation loss, perplexity, task metric, learning rate, gradient norm, improvement rate);
  `analysis_summary.json` + `training_curve_report.md`. Consumes analysis data only.
- `scripts/analyze_training_curve.py` — CLI coordinating load → analyze → report → plot with
  all requested flags (`--future-step` repeatable, `--plot`, `--plot-dir`,
  `--plot-format png|svg|both`, `--show-confidence`, `--log-x`, `--log-y`, `--target-loss`, …).
- `docs/training_curve_analysis.md` — feature docs + explicit limitations.
- `tests/test_curve_analysis.py` (20) and `tests/test_curve_plots.py` (10).
Files modified (no architecture change): `scripts/pretrain_mlm.py` — added `--metrics-file`
JSONL/CSV logging (with gradient-norm capture) and `--early-stop-policy off|warn|stop`
(default off; stop is guarded + always checkpoints first).

Commands run (all inspected):
- `python3 -m pytest` → **65 passed, 1 xfailed**.
- Example: `analyze_training_curve.py --metrics experiments/run_001/metrics.jsonl --run-id
  run_001 --future-step 1000 --future-step 2000 --plot --show-confidence` → status CONTINUE,
  power fit (R²≈0.97), asymptote ≈1.687, forecasts + CIs, 6 figures, summary + report.
  Validation-loss figure visually inspected and correct.
- Smoke run with `--metrics-file … --early-stop-policy warn` → metrics logged, analyzer ran
  each eval, did not auto-stop, final checkpoint saved. See experiment_log.md EXP-003.

Findings / honesty notes:
- scipy is not installed here, so the **numpy grid+least-squares fitting path** is the one
  actually exercised (the intended fallback). LAPACK `DLASCL` noise from degenerate bases
  (e.g. step 0 → `0**-α`) was fixed by cleaning to strictly-positive finite points and
  guarding `lstsq` inputs. A lone gradient spike among identical values (MAD=0) was missed
  until a median-multiple rule was added alongside the robust-scale threshold.
- Design keeps analysis / plotting / CLI separate (requirement) — analysis returns structured
  data; plotting and the report are pure consumers; the JSON summary matches the recommendation.

Limitations recorded (docs + report disclaimer + ADR-009): heuristic extrapolation only, no ML
forecaster, no optimality claim; unreliable after phase transitions / instability; perplexity
is not the task; the target "probability" is an uncalibrated bootstrap fraction; auto-stop is
opt-in and conservative.

No `src/coordinator_bert/{model,attention,embeddings,mlm_head,masking,data,checkpointing}.py`
changes. Next step unchanged: BF16 on DGX Spark, then Milestone 1 (100M).

---

## 2026-07-11 — Pre-GitHub deployment prep: PHASE 1 (inspect & preserve)

Pre-change verified state recorded before any deployment-prep edits.

- `git status`: single commit `52fe1c0` on `master`; all Milestone 0.5/0.6 files are present in
  the working tree but **uncommitted** (the earlier `.git/HEAD.lock` on this synced mount
  blocked follow-up commits — noted for Phase 9). `.venv/`, `.pytest_cache/`, `.DS_Store` are
  already git-ignored.
- `git log --oneline --decorate -10`: `52fe1c0 (HEAD -> master) Milestone 0: custom 25M BERT
  MLM system (model, data, training, tests)`.
- `python3 --version`: Python 3.10.12 (system). A synced `.venv/` (python3.11) exists but its
  interpreter symlink is dangling in this Linux sandbox; system python3.10 with the installed
  packages is used instead.
- `python -m pytest -q`: **65 passed, 1 xfailed** (the intentional distillation-placeholder
  xfail). This is the verified baseline to preserve.
- Execution environment reality check: this sandbox is **Linux aarch64, CPU-only** (no MPS, no
  CUDA). It stands in for the "Mac local" environment and resolves to CPU/fp32. MPS and CUDA
  code paths are written with feature detection and unit-tested via mocks, but cannot be
  exercised on real Apple/NVIDIA hardware here. Reports will claim **DGX readiness, not DGX
  validation**, per the task.

Preserved functionality (all currently green): custom BERT, 65 tests + 1 xfail, MLM training,
checkpoint save/resume, inference utilities, tiny overfit test, synthetic evaluation, metrics
logging, learning-curve analysis, and static graphs + Markdown/JSON reports. No model
architecture rewrite is planned — changes are packaging, config, diagnostics, benchmarking,
checkpoint hygiene, git hygiene, docs, and validation.

Stray untracked experiment dirs observed (to be git-ignored in Phase 9): `experiments/run_001`,
`experiments/_atc_preview`, `experiments/eval_ckpt_probe`, loose `experiments/smoke/metrics_*.jsonl`.

---

## 2026-07-11 — Pre-GitHub release prep: PHASES 2–13 complete

Converted the working repo into a dual-platform (Mac MPS / DGX CUDA) release candidate. No
model-architecture rewrite. Verified baseline preserved and extended: **88 passed, 1 xfailed**.

Files added:
- `src/coordinator_bert/runtime.py` — device/precision resolution + feature detection + safe
  fallbacks + startup report (Phase 6).
- `configs/model/*`, `configs/platform/*`, `configs/experiments/*`, `configs/examples/minimal.yaml`,
  and resolved `configs/bert_25m_{mac,dgx_portability,dgx_throughput}.yaml`,
  `configs/bert_{100m,200m}_dgx.yaml` (Phase 3).
- `scripts/check_environment.py` (Phase 4), `scripts/benchmark_training.py` (Phase 5).
- `docs/DGX_DEPLOYMENT.md`, `docs/RELEASE_CHECKLIST.md` (Phases 10–11).
- `tests/test_runtime.py` (12), `tests/test_checkpoint_manager.py` (9),
  `tests/test_curve_fixture.py` (2), `tests/fixtures/curve_metrics.jsonl` (tracked fixture).
- `src/coordinator_bert/py.typed`, `experiments/.gitkeep`.

Files modified:
- `pyproject.toml` — minimal core + extras `dev/train/analysis/scipy_optional/wandb/all`
  (Phase 2).
- `src/coordinator_bert/configuration.py` — `RuntimeConfig`, `extends` composition, `to_dict`.
- `src/coordinator_bert/checkpointing.py` — atomic writes, SHA-256 metadata, verify-on-load,
  immutable `CheckpointManager` + `latest.json`, `resolve_checkpoint_path` (Phase 8).
- `src/coordinator_bert/data.py` — runtime-aware DataLoader kwargs (workers/pin/persistent).
- `src/coordinator_bert/inference.py` — resolve checkpoint root via latest.json.
- `scripts/pretrain_mlm.py` — resolved-runtime integration (device/precision/TF32/fused/compile),
  runtime-aware dataloaders, run-artifact writing, immutable checkpoints + best pointer.
- `.gitignore` — comprehensive (Phase 9); `CLAUDE.md`, `README.md` — dual-platform docs + policy.

Mac (CPU stand-in) validation: install, env JSON, 88-test suite, smoke run, checksum-verified
resume, inference, synthetic eval, curve analysis + plots, and benchmark all pass. See
experiment_log.md EXP-004.

Honest limitations: MPS/CUDA/BF16 paths are feature-detected + mock-tested but NOT run on real
Apple/NVIDIA hardware — **DGX readiness, not validation**. The synced mount inflates checkpoint
I/O and blocks git commits via a stale `.git/HEAD.lock` (remediation in RELEASE_CHECKLIST.md).
Coordination heads, distillation, routing, voice remain placeholders (unchanged).

Next: on the real Mac — clear `.git/HEAD.lock`, `git add -A`, `git rm --cached
experiments/smoke/smoke_train.log`, commit + tag `v0.1.0-rc1`, push. Then follow
`docs/DGX_DEPLOYMENT.md` on the DGX Spark.

---

## 2026-07-11 — Optional W&B experiment tracking (no math changes; local pipeline intact)

Added optional Weights & Biases tracking. Default backend `none` (no-op); training byte-
identical when off. **105 passed, 1 xfailed** (+17 tracking tests, mocked wandb — no network).

Files added: `src/coordinator_bert/tracking.py` (Null/Wandb backends, lazy wandb import, secret
redaction, offline sync command), `tests/test_tracking.py`, `docs/WANDB_INTEGRATION.md`.
Files modified: `pyproject.toml` (wandb extra already present), `configuration.py`
(`TrackingConfig` + RunConfig field), `__init__.py`, `scripts/pretrain_mlm.py` (build tracker,
init with run identity, log `train/eval/system/analysis/*` with global_step, summary, optional
typed artifacts, finally-finish, `--wandb*` CLI flags), platform/example configs (DGX default
wandb/offline, Mac none), `README.md`, `docs/RELEASE_CHECKLIST.md`.

Design (ADR-013): backend-agnostic interface; no top-level wandb import; `finish()` in a
`finally` block always runs (incl. exceptions); offline needs no auth/network and never auto-
syncs; secrets redacted from config/summary; checkpoints not uploaded unless `log_checkpoints`;
local JSONL + curve analysis remain the source of truth (analyzer status also logged as
`analysis/*`, not replaced).

Verified: install with wandb extra; full suite; backend-none regression; live **offline** run
(auto run name, local run dir, sync command printed, not auto-synced, metrics.jsonl + config +
environment + checkpoint all present) — all with **no network**. See experiment_log EXP-005.

Honest limits: DGX/online W&B not exercised on real hardware/account — offline path validated on
CPU only. No secrets in repo; online auth uses the W&B CLI / `WANDB_API_KEY` env var, never read
or committed by this project.
