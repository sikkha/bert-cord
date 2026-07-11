# Training-curve analysis (heuristic learning-curve extrapolation)

`scripts/analyze_training_curve.py` estimates whether a run is **still improving enough to
justify more compute**, and renders static research figures. It is intentionally conservative.

## What this is — and is NOT

- This is **heuristic learning-curve extrapolation**: a few simple closed-form curves fit to
  past validation loss, with bootstrap uncertainty.
- It is **not** a learned forecasting model, and there is deliberately **no ML forecaster**.
- **AdamW does not predict the optimal stopping point.** Nothing here knows the true optimum.
- **Forecasts are unreliable after phase transitions or optimizer instability** (LR warmup
  end, schedule changes, loss spikes, NaN/Inf). When instability is detected, forecasts are
  suppressed and the status becomes `UNSTABLE`.
- **Perplexity/loss is not the task.** Loss can fall while the task metric (e.g. masked-token
  accuracy) stays flat. Always weigh task-specific metrics alongside perplexity — the task
  figure is annotated when loss improves but accuracy does not.
- **No claim of scientific optimality** is made. Treat outputs as a rough, honest guide.

## Inputs

A CSV or JSONL metrics file with per-evaluation rows. Recognized fields:

```
step, tokens_seen, train_loss, val_loss, masked_accuracy, learning_rate, gradient_norm
```

Missing fields are handled gracefully (e.g. `gradient_norm` is optional; non-eval rows may omit
`val_loss`). Only rows with a finite `val_loss` drive the loss analysis. A *reported* NaN/Inf
(the key is present but non-finite) is treated as instability; a merely-missing field is not.

## What it computes (analysis, `curve_analysis.py`)

- EMA-smoothed validation loss.
- Recent slope over both `step` and `log(step)`, and improvement per 100 steps.
- **Plateau** detection (configurable `patience`, `min_delta`, `min_evals`): best smoothed
  loss has not improved by `min_delta` for `patience` evaluations.
- **Instability** detection: reported NaN/Inf, sudden validation-loss spikes (robust MAD scale,
  high sigma, plus a relative floor so ordinary noise is not flagged), and gradient-norm
  excursions.
- Three candidate fits (fit only when enough finite points exist):
  - `L(t) = L_inf + A · t^(-alpha)` (power)
  - `L(t) = L_inf + A · exp(-k·t)` (exponential)
  - `L(t) = a + b / sqrt(t)` (inverse-sqrt)
  Nonlinear parameters are found by a bounded grid search; the linear coefficients by least
  squares (scipy is used only if already installed — a numpy path is the default). Models are
  compared by **held-out tail RMSE** (fit early, test on the recent tail), with **AIC** as a
  fallback.
- Forecasts at requested future steps, estimated asymptotic loss, and **bootstrap confidence
  intervals** (resampling `(step, loss)` pairs). If too few bootstrap refits succeed the CI is
  reported as `NaN`, not a false-precision number.
- A **probability-like heuristic** that further training beats a `--target-loss` — the fraction
  of bootstrap forecasts at or below the target. **Not a calibrated probability.**
- A single status: `CONTINUE` / `PLATEAU` / `UNSTABLE` / `INSUFFICIENT_DATA`.

Analysis functions return structured data only; plotting and the CLI consume that data.

## Status precedence

1. Reported NaN/Inf → `UNSTABLE`.
2. Fewer than `min_evals` evaluations → `INSUFFICIENT_DATA`.
3. Loss/gradient spikes → `UNSTABLE`.
4. Plateau with negligible predicted gain → `PLATEAU`.
5. Otherwise → `CONTINUE`.

## Figures (plotting, `curve_plots.py`, matplotlib only, headless `Agg`)

Separate PNG/SVG files (no subplots, no seaborn, no display required):

- **A. validation_loss_curve** — raw points, EMA, best point, plateau region, instability
  marks, fitted forecast, bootstrap CI, current-step and recommended-stop markers.
- **B. perplexity_curve** — `perplexity = exp(loss)`, forecast + CI, auto log-y for wide ranges.
- **C. task_metric_curve** — masked-token accuracy (raw + smoothed), chance baseline, best
  value; forecast omitted (not statistically justified); annotated if loss improves but the
  task metric is flat.
- **D. learning_rate_curve** — LR by step with warmup→decay boundary and instability marks.
- **E. gradient_norm_curve** — only if present; raw + smoothed, anomaly threshold, spike marks.
- **F. improvement_rate_curve** — Δloss per 100 steps, zero reference, negligible-gain band,
  and forecast improvement over the remaining budget.

Plotting correctness: missing fields are skipped (not errored); lines break across NaN gaps and
are never interpolated as continuous; forecasts appear only when a fit exists; observed and
forecast regions are visually distinct; axes are labelled with units; titles include the run
identifier; no misleading truncated axes.

Outputs also include a machine-readable `analysis_summary.json` (status, steps, forecasts, CIs,
chosen model, fit quality, warnings) and a concise `training_curve_report.md` that links the
figures and explains the recommendation. The JSON summary always matches the plotted / reported
recommendation.

## CLI

```bash
python scripts/analyze_training_curve.py \
  --metrics experiments/run_001/metrics.jsonl \
  --future-step 1000 \
  --future-step 2000 \
  --plot \
  --plot-dir experiments/run_001/analysis \
  --show-confidence
```

Flags: `--future-step` (repeatable), `--target-loss`, `--patience`, `--min-delta`,
`--min-evals`, `--min-fit-points`, `--ema-alpha`, `--slope-window`, `--negligible-gain-per-100`,
`--n-boot`, `--ci`, `--out-dir`; plotting: `--plot`, `--plot-dir`, `--plot-format png|svg|both`,
`--show-confidence`, `--log-x`, `--log-y`, `--chance-baseline`.

This first version is static, reproducible figures only — no interactive dashboard, no W&B
integration.

## Optional trainer integration (`scripts/pretrain_mlm.py`)

- `--metrics-file PATH` writes per-eval metrics as JSONL (or CSV by extension). Includes
  `gradient_norm` captured from gradient clipping.
- `--early-stop-policy off|warn|stop` — **default `off`** (no analysis, never stops):
  - `warn`: prints the analyzer's recommendation at each evaluation; never stops.
  - `stop`: may stop, but **only** after all guards hold — minimum evaluations reached,
    patience exceeded, **no instability event**, and predicted gain below
    `--es-predicted-gain`. A **final checkpoint is always saved before stopping**.

Early-stop tuning flags: `--es-patience`, `--es-min-delta`, `--es-min-evals`,
`--es-min-gain-per-100`, `--es-predicted-gain`, `--es-future-step`.

Because AdamW does not know the optimum, `stop` is opt-in and conservative by design; `warn` is
the recommended way to keep a human in the loop.
