# Experiment Log

Every smoke/training run, with actual inspected output. No run is marked successful without
inspecting its output.

---

## EXP-001 — Milestone 0 MLM smoke training (staged run + resume)

- Config: `configs/bert_25m.yaml` with `--smoke` overrides (max_seq_length 32,
  per_device_batch_size 8, gradient_accumulation_steps 2, lr 5e-4, warmup 4, cosine decay,
  periodic saves off, final `last` checkpoint).
- Git commit: `52fe1c0` (code state at run time).
- Hardware: Linux 6.8.0 aarch64, CPU-only (no CUDA), 4 CPU threads, 3.9 GB RAM, no swap.
- Precision: **fp32** (bf16 unavailable — no CUDA; reported honestly at startup).
- Model: custom `BertForMaskedLM`, **27,010,304 params (~27.01M)**, tied embeddings.
- Dataset: synthetic learnable copy-motif corpus (period 3, 64-token motif sub-vocab),
  256 train / 64 val examples; dynamic 15% / 80-10-10 masking.

### Stage A — fresh training, steps 0 → 20

Command:
```
OMP_NUM_THREADS=4 python3 scripts/pretrain_mlm.py --config configs/bert_25m.yaml \
    --smoke --max-steps 20
```
Inspected output (selected):
```
step  2 | loss 10.4093 | lr 2.500e-04
step  4 | loss 10.2799 | lr 5.000e-04
[eval] step 5  | val_loss 9.6818 | masked_acc 0.0213 | ppl 16022.77
step 10 | loss 8.7760
[eval] step 10 | val_loss 8.7946 | masked_acc 0.0110 | ppl 6598.63
[eval] step 15 | val_loss 7.9723 | masked_acc 0.0000 | ppl 2899.66
step 20 | loss 7.7584
[eval] step 20 | val_loss 7.6713 | masked_acc 0.0000 | ppl 2145.88
[checkpoint] saved step 20 -> experiments/smoke/checkpoints/last
[final] step 20 | val_loss 7.6485 | ppl 2097.45
[final] elapsed 11.9s | 757 tok/s | peak_mem 1166.2 MB | tokens_seen 9,010
```
Result: exit 0. Train loss 10.41 → 7.76; val loss 9.68 → 7.65. Checkpoint written at
`global_step=20`.

### Stage B — resume from checkpoint, steps 20 → 40

Command:
```
OMP_NUM_THREADS=4 python3 scripts/pretrain_mlm.py --config configs/bert_25m.yaml \
    --smoke --max-steps 40 --resume experiments/smoke/checkpoints/last
```
Inspected output (selected):
```
[resume] restored global_step=20 from experiments/smoke/checkpoints/last
step 22 | loss 7.7040
[eval] step 25 | val_loss 7.1218 | masked_acc 0.0106 | ppl 1238.72
[eval] step 30 | val_loss 6.6925 | masked_acc 0.0110 | ppl 806.35
[eval] step 35 | val_loss 6.3377 | ppl 565.48
step 40 | loss 6.1105
[eval] step 40 | val_loss 6.1416 | masked_acc 0.0000 | ppl 464.80
[checkpoint] saved step 40 -> experiments/smoke/checkpoints/last
[final] step 40 | val_loss 6.1114 | ppl 450.97
[final] elapsed 12.5s | 724 tok/s | peak_mem 1326.1 MB | tokens_seen 9,038
```
Result: exit 0. Resume restored step 20; step 22 loss 7.70 continues smoothly from stage A's
7.76 (no optimizer-state discontinuity/spike). Loss fell further to val 6.11 at step 40.

### Metrics summary

| metric              | value                                            |
|---------------------|--------------------------------------------------|
| runtime (per stage) | ~12 s training each (fp32, CPU)                  |
| throughput          | ~720–760 tokens/s                                |
| peak RSS            | ~1.1–1.3 GB                                       |
| train loss          | 10.41 → 6.11 (across both stages)                |
| val loss            | 9.68 → 6.11                                       |
| val perplexity      | 16023 → 451                                       |
| masked-token acc    | ~0–2% (coarse: ~100 masked eval tokens; top-1    |
|                     | over 32000-vocab is hard this early)             |
| checkpoint (final)  | `experiments/smoke/checkpoints/last` (step 40)   |

### Interpretation

The gradient/optimizer/scheduler path works end-to-end: loss decreases monotonically on both
train and val, and resume continues without a loss spike, confirming optimizer and scheduler
state were correctly restored. Masked-token accuracy is *valid but low/coarse* at this tiny
scale — the eval set exposes only ~100 masked tokens and top-1 accuracy over the full vocab
is a hard, high-variance metric after ~40 tiny steps; loss/perplexity are the reliable signals
here. No NaNs, no runtime errors, no unsupported-precision claims.

### Caveats / honesty notes

- CPU/fp32 only in this environment; the **CUDA BF16 path was not exercised on real hardware**
  and remains to be validated on the DGX Spark.
- Smoke uses a synthetic learnable corpus, not real text; it validates the *pipeline*, not
  language modeling quality.
- Two stale checkpoint dirs (`step_5`, `step_20`) remain from earlier timed-out runs; this
  mount disallows `unlink`, but they are git-ignored and harmless. `last` is authoritative.

---

## Reproducibility record (per brief)

```
seed: 42
config: configs/bert_25m.yaml (+ --smoke overrides)
torch: 2.13.0+cpu
cuda: None (CPU)
device: cpu
precision: fp32
dataset: synthetic copy-motif (period=3, motif_vocab=64), seeds 42/43
tokenizer: synthetic reserved specials (PAD0 CLS1 SEP2 MASK3 UNK4); real vocab from id 5
git commit: 52fe1c0
```

---

## EXP-002 — Milestone 0.5 evaluation utilities (inference / overfit / synthetic gen)

Date: 2026-07-11. Hardware/precision identical to EXP-001 (Linux aarch64, CPU-only, fp32,
4 threads). Model: custom `BertForMaskedLM`, 27,010,304 params. All outputs below were
inspected directly; no result is reported that was not observed.

New code: `src/coordinator_bert/inference.py` (helpers), `scripts/predict_mask.py`,
`scripts/overfit_tiny.py`, `scripts/evaluate_synthetic.py`, `tests/test_inference.py`.
Scope: validate inference mechanics, overfitting *capacity*, and synthetic *generalization
measurement* only. **No claim of language understanding is made.**

### Test suite (incl. new inference tests)

Command: `python3 -m pytest`
Inspected result: **`35 passed, 1 xfailed in 1.43s`** (was 26+1 in EXP-001; +9 inference
tests: checkpoint load restores weights, eval-mode determinism, top-k output shape,
masked-position extraction, empty-mask handling, top-k accuracy incl. ignore_index).

### overfit_tiny.py — capacity check (headline result)

Command: `python3 scripts/overfit_tiny.py --config configs/bert_25m.yaml`
(defaults: 8 seqs × 16 tok, period 3, 13 masked positions, lr 1e-3, warmup 20, grad-clip 1.0,
80 steps, threshold 0.90)
Inspected output:
```
step   1 | loss 10.6015 | top1 0.000 | top5 0.000
step  25 | loss  0.1099 | top1 1.000 | top5 1.000
step  50 | loss  0.0006 | top1 1.000 | top5 1.000
[overfit] final loss 0.0002 | masked top1 1.0000 | masked top5 1.0000 | 16.9s
[overfit] PASS: top1 1.0000 > threshold 0.9
```
Result: **PASS**, exit 0. The model drives a fixed tiny batch to loss ≈ 0 and masked top-1 =
1.000 — the gradient/optimizer path has full capacity to fit data.

Note (recorded honestly): an initial attempt with lr 2e-3 and **no warmup / no grad-clip**
plateaued at loss ≈ 2.27, top-1 ≈ 0.15 (post-LN instability, loss bouncing). Adding a 20-step
linear warmup and gradient clipping (max-norm 1.0) fixed it — both are now defaults in the
script. This is itself a useful finding about post-LN training stability at this scale.

### evaluate_synthetic.py — synthetic generalization measurement

Command (baseline): `python3 scripts/evaluate_synthetic.py --config configs/bert_25m.yaml
--periods 2 3 4 --seq-lens 24 48 --num-examples 32` (no checkpoint → random init)
Inspected output (deterministic): every combo **top-1 = 0.0000, top-5 = 0.0000**, loss ≈ 10.45
(≈ ln 32000), overall top1 0.0000. Confirms the metric's zero-point.

Command (lightly-trained): same, with `--checkpoint experiments/smoke/checkpoints/last`
against the EXP-001 ~40-step smoke checkpoint. Inspected output:
```
period seq_len    loss     top1    top5   masked
   2      24     9.617    0.067   0.076    105
   2      48     9.700    0.038   0.062    208
   3      24     9.872    0.010   0.052     97
   3      48     9.880    0.009   0.057    212
   4      24     9.888    0.000   0.037    107
   4      48     9.832    0.037   0.048    187
overall: loss 9.798 | top1 0.0271 | top5 0.0554 over 6 combos
```
Interpretation: the utility works and cleanly discriminates a lightly-trained model
(top-1 ≈ 2.7%, top-5 ≈ 5.5%, loss ≈ 9.80) from random init (0.0 / loss ≈ 10.45). The absolute
numbers are low because that checkpoint saw only ~40 tiny steps — the point here is that the
*measurement* is correct and above-baseline, not that the model is good.

Observation on training-to-generalize: a separate in-process attempt to train a fresh model
(period-mixed data, 60 steps, warmup+clip) plateaued at loss ≈ 4.18 ≈ ln(64) — it learned to
restrict predictions to the 64-token motif sub-vocabulary but had not yet learned the exact
positional-copy rule in so few CPU-bounded steps. Overfit capacity (EXP-002 above) is proven;
strong synthetic *generalization* would need more training steps than the 45s-per-call CPU
budget here allows. Deferred to GPU (DGX Spark).

### predict_mask.py — top-k masked prediction (inference mechanics)

Command: `python3 scripts/predict_mask.py --config configs/bert_25m.yaml
--checkpoint experiments/smoke/checkpoints/last --period 3 --seq-len 24 --topk 5`
Inspected output: correctly generated a period-3 sequence
`[1,50,67,7,50,67,7,...,2]`, masked position 12 (true id 7), loaded the checkpoint, and printed
5 ranked (id, probability) candidates. Mechanics verified end-to-end. Predictions are poor /
near-uniform (top-1 p≈0.0003) because the smoke checkpoint is barely trained — expected and
consistent with evaluate_synthetic. (Explicit `--input` token-id mode and default-middle
masking are also implemented and unit-mirrored in tests.)

### Caveat — checkpoint persistence on this synced mount

The workspace folder is a synced mount that reliably supports **creating new files** but is
**unreliable at persisting overwrites of large binary files across separate runs**. Concretely,
`experiments/smoke/checkpoints/last/state.pt` did not consistently reflect the latest step
between shell invocations (its `meta.json` was observed at step 4, then 40, then 999 after an
overwrite probe). This did not affect any single-process result above (each script loads and
uses the checkpoint within one process). It does mean a *persisted* well-trained checkpoint
could not be reliably produced here; on a normal filesystem (and on the DGX Spark) this is a
non-issue. The reliable, reproducible M0.5 evidence is: overfit_tiny top-1 = 1.000 (capacity)
and evaluate_synthetic random-init top-1 = 0.000 (metric zero-point), both fully in-process.

### Reproducibility (EXP-002)

```
torch 2.13.0+cpu | cuda None | device cpu | precision fp32
overfit_tiny: seed 42, lr 1e-3, warmup 20, grad-clip 1.0, 8x16 period-3 fixed batch
evaluate_synthetic: eval seed base 20260711 (disjoint from train seeds), periods {2,3,4},
                    seq_lens {24,48}, mlm_probability 0.15
predict_mask: seed 123, period 3, seq_len 24, top-k 5
tests: 35 passed, 1 xfailed
```

---

## EXP-003 — Milestone 0.6: training-curve analysis + visualization

Date: 2026-07-11. Environment identical (Linux aarch64, CPU, numpy 2.2.6, matplotlib 3.10.9,
**scipy not installed** → the numpy grid+lstsq fitting path was exercised). Feature adds
`src/coordinator_bert/curve_analysis.py`, `src/coordinator_bert/curve_plots.py`,
`scripts/analyze_training_curve.py`, trainer integration in `scripts/pretrain_mlm.py`, tests
`tests/test_curve_analysis.py` + `tests/test_curve_plots.py`, and
`docs/training_curve_analysis.md`. **Heuristic learning-curve extrapolation only — no ML
forecaster, no claim of optimality.**

### Tests (inspected)

`python3 -m pytest` → **`65 passed, 1 xfailed in 6.03s`** (was 35+1). New:
- `test_curve_analysis.py` (20): status classification — improving→CONTINUE, flat→PLATEAU,
  noisy-but-improving→CONTINUE, spike→UNSTABLE, NaN→UNSTABLE, Inf→UNSTABLE, gradient
  excursion→UNSTABLE, too-few→INSUFFICIENT_DATA; degenerate inputs (empty / single point /
  constant / all-NaN / zero step-spread / 1e300 / junk) never crash and stay JSON-serializable;
  CSV+JSONL loading; sparse val rows not mis-flagged as NaN; power-data fit recovers asymptote;
  prob-beats-target ∈ [0,1].
- `test_curve_plots.py` (10, headless): all six figures created for valid metrics; PNG+SVG both
  produced; flat curve embeds a "plateau region" annotation (checked in SVG text); unstable
  curve marks spikes; insufficient-data → observed-only, no "forecast" series; missing
  gradient_norm / task / LR fields skip those figures without error; `analysis_summary.json`
  status + recommended-stop match the analysis; Markdown report links figures; works with no
  DISPLAY.

### Example command (inspected, produced real artifacts)

```
python3 scripts/analyze_training_curve.py --metrics experiments/run_001/metrics.jsonl \
  --run-id run_001 --future-step 1000 --future-step 2000 \
  --plot --plot-dir experiments/run_001/analysis --show-confidence
```
On a synthetic 12-eval power-law run (to step 960): status **CONTINUE**, chosen model
**power** (R²≈0.97, tail RMSE≈0.03), estimated asymptote ≈ **1.687**, forecast @1000 ≈ 2.078
(90% CI [2.018, 2.133]), @2000 ≈ 1.979 (CI [1.874, 2.114]). Produced 6 figures + a widening
bootstrap CI band in the forecast region (validation-loss figure visually inspected: observed
points, EMA, best-point star, dotted fit over observed range, dashed forecast beyond the
current-step marker, asymptote line — observed/forecast regions clearly distinct), plus
`analysis_summary.json` and `training_curve_report.md`.

### Trainer integration (inspected)

Smoke run with `--metrics-file … --early-stop-policy warn --es-min-evals 3 --es-patience 3`:
per-eval JSONL rows written (incl. `gradient_norm` captured from grad clipping), analyzer ran
each eval printing status (INSUFFICIENT_DATA → CONTINUE), **did not auto-stop** (`warn` mode;
`early_stopped=False`), final checkpoint saved. Default `--early-stop-policy off` leaves
training behaviour unchanged.

### Honest limitations (also in docs + report disclaimer)

Forecasts are heuristic and unreliable after phase transitions / optimizer instability
(suppressed under `UNSTABLE`). Perplexity is not the task — the task-metric figure is annotated
when loss improves while masked accuracy stays flat. The target "probability" is a bootstrap
fraction, not calibrated. `stop` policy is opt-in and gated on no-instability + low predicted
gain, and always checkpoints first.

### Reproducibility (EXP-003)

```
numpy 2.2.6 | matplotlib 3.10.9 (Agg) | scipy: not installed (numpy fit path used)
analysis defaults: ema_alpha 0.3, patience 5, min_delta 1e-3, min_evals 6, min_fit_points 6,
                   n_boot 200, ci 0.90, bootstrap_seed 0
tests: 65 passed, 1 xfailed
```

---

## EXP-004 — Pre-GitHub release prep: Mac (CPU stand-in) validation

Date: 2026-07-11. Environment: Linux aarch64, CPU-only, fp32 (stands in for the Mac local env;
resolves device=cpu). MPS/CUDA paths are feature-detected + unit-tested via mocks but not run
on real hardware here. **DGX readiness only — not DGX validation.**

Validation sequence (all inspected):

1. Install: `pip install -e ".[dev,train,analysis]"` — success; all extras import; scipy absent
   (numpy curve-fit path used).
2. Env report: `check_environment.py --json …` → device cpu, precision fp32, ok=True;
   `--require training` exit 0; `--require dgx` exit 1 (no CUDA — correct off-DGX).
3. Tests: `python -m pytest` → **88 passed, 1 xfailed** (was 65+1; +12 runtime, +9 checkpoint
   manager, +2 curve fixture; 1 fixture assertion fixed re: EMA-smoothed best).
4. Smoke run: `pretrain_mlm.py --config configs/bert_25m_mac.yaml --smoke --max-steps 20
   --metrics-file …` → device cpu / precision fp32, resolved-runtime printed, 10 metrics rows,
   immutable checkpoint `step_000020` + `latest.json`, val_loss 7.65.
5. Resume: `--max-steps 30 --resume experiments/smoke/checkpoints` → followed latest.json,
   **checksum verified**, restored step 20, continued to step 30 (val 7.18), wrote immutable
   `step_000030` (step_000020 left immutable).
6. Inference: `predict_mask.py --checkpoint experiments/smoke/checkpoints` (root → resolved
   latest) → top-k printed (weak 30-step checkpoint; predictions near-uniform, expected).
7. Synthetic eval: `evaluate_synthetic.py` → overall loss 7.07, top5 0.078 (above random).
8. Curve analysis: `analyze_training_curve.py --plot` → status CONTINUE, 6 figures +
   `analysis_summary.json` + `training_curve_report.md` written.
9. Benchmark: `benchmark_training.py --config configs/experiments/smoke_mac.yaml --steps 12
   --eval --checkpoint --probe-batch-sizes --probe-candidates 8 16 32` → steps/s 3.03,
   tokens/s 692, median latency 256 ms, checkpoint overhead measured, batch probe largest=32,
   all outputs (environment.json, resolved_config.yaml, metrics.jsonl, benchmark_summary.json,
   benchmark_report.md, plots/) produced. (The batch16/seq128 portability-config benchmark is
   too slow on this CPU box within the 45 s call limit — run it on the actual Mac/DGX.)

Notes / honesty: checkpoint save overhead is inflated (~12 s for 323 MB) by this slow, nearly
full synced mount; real disks are far faster. The synced mount also blocks git commits/index
writes via a stale `.git/HEAD.lock` (Operation not permitted) — remediation documented for the
Mac. No NaN/Inf, no runtime errors, no unsupported-precision claims.

### Reproducibility (EXP-004)

```
torch 2.13.0+cpu | numpy 2.2.6 | matplotlib 3.10.9 | scipy: absent
device cpu | precision fp32 | seed 42
tests: 88 passed, 1 xfailed
actual parameter count: 27,010,304 (~27.01M)
```

---

## EXP-005 — Optional W&B tracking (offline verification)

Date: 2026-07-11. Environment: Linux aarch64, CPU, wandb 0.28.0 installed. **No network used.**

New: `src/coordinator_bert/tracking.py`, `TrackingConfig`, trainer integration (metrics/summary/
artifacts + finally-finish), `tests/test_tracking.py` (17, mocked wandb), tracking sections in
platform/example configs, `docs/WANDB_INTEGRATION.md`.

Verification (all inspected):

1. `pip install -e ".[dev,train,analysis,wandb]"` — success; project still imports/runs without
   wandb (NullTracker is the default).
2. Tests: `python -m pytest` → **105 passed, 1 xfailed** (was 88+1; +17 tracking). Mocked wandb;
   no test contacts wandb.ai.
3. Backend `none` regression: smoke run identical to before, no tracking output.
4. **Offline run** (`--config … tracking.backend=wandb mode=offline`, `--max-steps 6
   --eval-every 3`): auto run name `bert27m-cpu-wb_verify-20260711-185240`; local W&B dir
   `experiments/wb_verify/wandb/offline-run-*` created **without auth/network**; printed
   `wandb sync <dir>` and "offline runs are NOT synced automatically". Confirmed present
   afterward: `metrics.jsonl` (2 eval rows), `resolved_config.yaml`, `environment.json`,
   `checkpoints/latest.json` — the local pipeline is intact.
5. Secret handling: config/summary redaction unit-tested; a scan of the offline run found no
   real secrets (only the package name `SecretStorage` in `requirements.txt`).

Notes: metrics.jsonl is written only when an eval occurs (needs `eval_every ≤ steps`). Checkpoint
save overhead remains inflated by the slow synced mount. Training math unchanged. **DGX W&B not
run on real hardware — offline path validated on CPU only.**

### Reproducibility (EXP-005)

```
wandb 0.28.0 | mode offline | no network | no login
tests: 105 passed, 1 xfailed
tracking default backend: none (no-op)
```

---

## EXP-006 — Milestone 0.7: ONNX export + parity (actual 27.01M model)

Date: 2026-07-11. Hardware: Linux aarch64, CPU-only. Precision: FP32. Packages: torch
2.13.0+cpu, onnx 1.22.0, onnxruntime 1.23.2, onnxscript 0.7.1. ORT provider:
`CPUExecutionProvider` (available: Azure, CPU).

- Config: `configs/bert_25m_mac.yaml` (model = 27.01M).
- Checkpoint: `experiments/smoke/checkpoints` → resolved via `latest.json` to `step_000040`.
- Export command:
  `python scripts/export_onnx.py --config configs/bert_25m_mac.yaml
  --checkpoint experiments/smoke/checkpoints --output exports/bert_cord_27m_mlm.onnx
  --sequence-length 128`
- **ONNX file size:** graph `bert_cord_27m_mlm.onnx` = 703,153 B (~0.69 MB); external weights
  `bert_cord_27m_mlm.onnx.data` = 107,479,040 B (~102.5 MB); **total ≈ 108,182,193 B (103.17
  MB)**. (Exporter externalizes FP32 weights; both files ship together.)
- **ONNX checker:** PASSED (structural validation).
- Opset: 18. Exporter: torch 2.13 dynamo path via `torch.onnx.export(dynamic_axes=…)`.
- Parity command:
  `python scripts/validate_onnx.py --config configs/bert_25m_mac.yaml
  --checkpoint experiments/smoke/checkpoints --onnx-model exports/bert_cord_27m_mlm.onnx
  --seq-lengths 128 64 --batch-sizes 1 3`
- **Test input shapes:** (1,128), (3,128,pad2), (1,64,pad2), (3,64) → outputs `[b, s, 32000]`.
- **Numerical parity:** `max|Δ| = 7.15e-6 … 8.11e-6`, `mean|Δ| ≈ 1.0e-6` (tolerances rtol=1e-3,
  atol=2e-3 → passed by ~3 orders of magnitude). No NaN/Inf.
- **Top-k agreement:** top-5 = **1.00** on all masked positions across all cases.
- Dynamic axes: exercised across seq {64,128} × batch {1,3} → OK.
- ORT inference: `predict_mask_onnx.py` ran at seq 24 and seq 48 (dynamic), reported providers,
  output `[1,24,32000]` / `[1,48,32000]`; malformed mask position → clear error, exit 2.
- Application-level parity: PyTorch `predict_mask.py` top-1 (id=62, p=0.0241) == ONNX top-1
  (id=62, p=0.0241) on the same input.
- Tests: **122 passed, 1 xfailed** (105+17 new ONNX; ORT-dependent tests skip if packages
  absent). Baseline before ONNX work: 105 passed, 1 xfailed.

Interpretation: the exported ONNX graph reproduces PyTorch MLM logits to FP32 numerical noise
with identical top-k, under dynamic batch/sequence, on ONNX Runtime CPU. The artifact is
inference-only (no training state).

Limitations: **FP32 CPU only** was validated. Apple **CoreML** EP and NVIDIA
**onnxruntime-gpu / CUDAExecutionProvider** were **not** run — untested. Checkpoint save/large
external-data write inflated by the slow synced mount; not a code issue.

---

## EXP-007 — Hugging Face ONNX package (local staging; no upload)

Date: 2026-07-11 (UTC). Hardware: Linux aarch64, CPU-only, FP32. onnx 1.22.0 / onnxruntime
1.23.2. **No network, no HF auth, no upload.**

- Package version: `0.1.1-onnx`; future HF repo: `sikkha/bert-cord-27m-mlm-onnx`.
- Source: repo `https://github.com/sikkha/bert-cord`, commit
  `0e17db558ebcce29f40b49d546af8b2704640230`, tag `v0.1.1-onnx` (detected automatically).
- Source ONNX: `exports/bert_cord_27m_mlm.onnx` (+ `.onnx.data`).
- Build command:
  `python scripts/build_hf_onnx_package.py --config configs/bert_25m_mac.yaml
  --onnx-model exports/bert_cord_27m_mlm.onnx --output bert-cord-27m-mlm-onnx
  --repo-id sikkha/bert-cord-27m-mlm-onnx --package-version 0.1.1-onnx`
- **Relink verified:** packaged `onnx/model.onnx` external-data location = `model.onnx.data`
  (re-saved, not renamed).
- Fresh parity (packaged ONNX vs PyTorch, same weights via `step_000040`): **max|Δ| = 8.11e-6**,
  top-5 agreement **1.00**, NaN/Inf **False**, provider `CPUExecutionProvider`, dynamic axes OK
  (batch {1,2,3} × seq {64,128}), onnx.checker PASS. (rtol=1e-3, atol=2e-3.)
- Package files + SHA-256 (from MANIFEST.json; created_utc 2026-07-11T16:42:16Z):
  - `LICENSE` 11,357 B `c71d239d…`
  - `README.md` 5,650 B `83a63b7b…`
  - `config.json` 847 B `27475a8e…`
  - `evaluation.json` 1,503 B `a910f611…`
  - `inference.py` 3,138 B `aa758aca…`
  - `requirements.txt` 30 B `2ec7b315…`
  - `onnx/model.onnx` 701,546 B `9f10b894…`
  - `onnx/model.onnx.data` 107,453,952 B `b234d8e4…`
  - Total ≈ 108,180,167 B (103.2 MB).
- Validation: `scripts/validate_hf_onnx_package.py bert-cord-27m-mlm-onnx` → **17/17 PASS**
  (required/forbidden files, JSON parse, README front matter, source-commit==HEAD, MANIFEST
  checksums, both ONNX files, external-data linkage, onnx.checker, ORT contract, dynamic
  batch+sequence, `inference.py` subprocess, no absolute-path leak, no secret leak).
- Standalone `python bert-cord-27m-mlm-onnx/inference.py` → providers `[CPUExecutionProvider]`,
  logits `(1, 24, 32000)`, top-5 ids `[62, 66, 23, 64, 39]` (matches PyTorch/ONNX top-1 62).
- Tests: **136 passed, 1 xfailed** (122 + 14 new HF-package tests, tiny fixtures, offline).

Interpretation: a self-contained HF model-repo package exists locally, honest and complete,
with verified external-data linkage and parity; ready for **manual** upload. Nothing was
uploaded or authenticated.

Limitations: synthetic MLM baseline; no tokenizer; CPU/FP32 validated only; CUDA/CoreML/FP16/BF16
unvalidated; not a coordinator. The staging dir is git-ignored (~103 MB).

### Later manual upload commands (NOT executed)

```
hf repo create sikkha/bert-cord-27m-mlm-onnx --type model
hf upload sikkha/bert-cord-27m-mlm-onnx bert-cord-27m-mlm-onnx .
```

---

## EXP-008 — HF package refinement (v0.1.2-hf-onnx): separated provenance + strict cleanup

Date: 2026-07-11. Linux aarch64, CPU/FP32. onnx 1.22.0 / onnxruntime 1.23.2. No network/HF auth.

- Separated provenance in config.json / evaluation.json / MANIFEST.json:
  `model_source_commit=0e17db55…`, `model_source_tag=v0.1.1-onnx` (ONNX export commit) vs
  `packaging_source_commit=0e17db55…`, `packaging_source_tag=v0.1.2-hf-package` (tooling commit).
  The distinct tags demonstrate the separation in the artifact.
- Package version bumped to `0.1.2-hf-onnx`.
- `rmtree(ignore_errors=True)` removed → strict `_prepare_output_dir`. Verified: building into the
  existing (unremovable, on this synced mount) `bert-cord-27m-mlm-onnx/` **aborts with exit 1** and
  the message "output directory … could not be removed cleanly … Remove it manually ('rm -rf') or
  pass a fresh --output path." — no mixed/stale package produced.
- Rebuilt from a **fresh path** `dist/bert-cord-27m-mlm-onnx` (dist/ git-ignored):
  `python scripts/build_hf_onnx_package.py --config configs/bert_25m_mac.yaml --onnx-model
  exports/bert_cord_27m_mlm.onnx --output dist/bert-cord-27m-mlm-onnx --repo-id
  sikkha/bert-cord-27m-mlm-onnx --package-version 0.1.2-hf-onnx --model-source-commit 0e17db55…
  --model-source-tag v0.1.1-onnx` → SUCCESS. Graph 701,546 B + weights 107,453,952 B (total
  108,180,892 B ≈ 103.2 MB). Parity max|Δ| 8.11e-6, top-5 1.00, no NaN/Inf, CPU.
- Validator (fresh package): **17/17 PASS** (now checks `packaging_source_commit == HEAD`).
- Standalone `inference.py`: logits (1,24,32000), top-5 ids [62,66,23,64,39].
- Tests: **139 passed, 1 xfailed** (+3 package tests: provenance separation; failed-cleanup abort;
  CLI non-zero exit on failed cleanup — via mocked `shutil.rmtree`).

Honest limitations (unchanged + new):
- **Git commit/tag could not be landed in this sandbox:** the synced mount re-created a
  `.git/index.lock` that cannot be unlinked ("Operation not permitted"), so `git add`/`git commit`
  fail here. HEAD remains `0e17db55`; the packaging tooling changes are on disk but uncommitted;
  the tag `v0.1.2-hf-package` exists pointing at `0e17db55`. On a normal filesystem:
  `rm -f .git/index.lock .git/HEAD.lock && git add -A && git commit -m "…" && git tag -a
  v0.1.2-hf-package -m "…"` completes it; then `packaging_source_commit` becomes the new hash.
- Still CPU/FP32 only; CUDA/CoreML/FP16/BF16 unvalidated. Not a coordinator; synthetic MLM baseline.

---

## EXP-009 — Tokenizer Milestone: pipeline validation on sample corpus

Date: 2026-07-13. Linux aarch64, CPU. tokenizers 0.23.1, datasets 5.0.0. Offline (no large
download). Goal: robust engineering pipeline, not tokenizer quality.

- Corpus prep: `prepare_tokenizer_corpus.py --input data/raw --output-dir data/tokenizer_corpus
  --val-fraction 0.1 --shard-size 100` on a ~2.6 KB hand-made multilingual sample (EN + Thai +
  markdown + jsonl/code). Result: 29 docs read/kept (0 dup on this set), languages {latin:22,
  thai:7}, 26 train docs in 1 shard + 3 validation; manifest + report written.
- Trained all three algorithms (vocab_size 32000 requested; tiny corpus → small actual vocab):
  | algo | actual vocab | special ids | note |
  |---|---:|---|---|
  | wordpiece | 361 | 0–4 OK | BERT normalizer |
  | byte_bpe  | 495 | 0–4 OK | byte-level, no UNK |
  | unigram   | 321 | 0–4 OK | Metaspace + byte fallback |
  Each wrote tokenizer.json + tokenizer_config.json + special_tokens_map.json +
  tokenizer_manifest.json + README.md; reserved-token integrity verified at train time.
- Evaluation (`evaluate_tokenizer.py` on validation + train):
  | algo | UNK rate | tok/sent | tok/word | round-trip (norm/exact) | vocab util | reserved |
  |---|---:|---:|---:|---|---:|---|
  | wordpiece | 0.71% | 29.1 | 4.50 | 58.6% / 58.6% | 73.7% | OK |
  | byte_bpe  | 0.00% | 32.4 | 5.00 | 100% / 0% | 55.2% | OK |
  | unigram   | 0.59% | 29.2 | 4.51 | 72.4% / 72.4% | 93.5% | OK |
  (byte-BPE: 0 UNK via byte fallback, full normalized round-trip; exact 0% because ByteLevel adds
  a leading space — expected.)
- Tests: **148 passed, 1 xfailed** (139 + 9 tokenizer tests, tiny fixtures, offline).
- HF reachability: the Hub responded, but `datasets 5.0.0` streaming hit a URI-parsing quirk for
  the bare `wikitext` id in this sandbox — noted in `docs/recommended_corpus.md`.

Corpus-size decision: all recommended real corpora (EN/TH Wikipedia, OSCAR, mC4, FineWeb, CC100)
**exceed 1 GB → not downloaded**; exact DGX download commands are in `docs/recommended_corpus.md`.

Interpretation: the prepare → train → evaluate pipeline is reproducible and correct on a small
offline corpus; all three algorithms honor the fixed special-token ids. A final algorithm will be
chosen and frozen after evaluation on the real corpus (on the DGX). No git commit performed
(left unstaged for manual review, per instructions).

Limitations: tiny sample corpus only; real 32k vocab needs the large corpus; tokenizer quality
not yet assessed on real multilingual text.
