# Weights & Biases integration (optional)

`bert_cord` can optionally mirror a run to Weights & Biases (W&B) for dashboards and Mac↔DGX
comparison. It is **entirely optional** and off by default.

## Purpose

- Visualize/compare runs across Mac and DGX without changing training math.
- Keep the project's **local JSONL metrics + static curve analysis as the source of truth** —
  W&B is an *addition*, never a replacement. Nothing here alters scientific results.

## Installation

```bash
python -m pip install -e ".[dev,train,analysis]"          # no W&B
python -m pip install -e ".[dev,train,analysis,wandb]"    # with W&B
```

The project installs and runs without `wandb`. Selecting the W&B backend while the package is
absent raises a clear, actionable error (`pip install -e ".[wandb]"`).

## Account / login (online mode only)

Offline mode needs **no account and no login**. For online mode, authenticate once with the
normal W&B mechanism:

```bash
wandb login            # prompts for your API key, stored by the W&B CLI (not by this repo)
# or: export WANDB_API_KEY=...   (never commit this; never put it in configs/dev_mem/git)
```

## Project creation

Set `tracking.project` (default `bert-cord`). W&B creates the project on first run under your
account/entity. Set `tracking.entity` for a team.

## Configuration

```yaml
tracking:
  backend: none          # none | wandb   (default none)
  mode: offline          # offline | online | disabled
  project: bert-cord
  entity: null
  run_name: null         # auto: <model>-<platform>-<experiment>-<timestamp>
  group: null
  job_type: training
  tags: []
  notes: null
  log_interval: 10
  log_code: false
  log_checkpoints: false        # do NOT upload checkpoints unless explicitly enabled
  log_analysis_artifacts: true  # config/env/metrics artifacts (small)
```

Or via CLI (overrides config): `--wandb`, `--wandb-mode {offline,online,disabled}`,
`--wandb-project NAME`, `--run-name NAME`, `--wandb-log-checkpoints`.

The DGX platform profiles default to `backend: wandb, mode: offline`; the Mac profile defaults
to `backend: none`.

## Offline mode (recommended default)

```bash
python scripts/pretrain_mlm.py --config configs/bert_25m_mac.yaml --smoke --max-steps 40 \
    --wandb --wandb-mode offline
```

- **No authentication or network required.**
- The offline run directory is created under `<output_dir>/wandb/offline-run-…` and printed at
  the end, together with the exact sync command.
- Runs are **never synced automatically**. Upload later, when you have network + login:

```bash
wandb sync <output_dir>/wandb/offline-run-YYYYMMDD_HHMMSS-<id>
```

## Online mode

Set `--wandb-mode online` (after `wandb login`). Requires network + a valid API key via the
standard W&B mechanism. Do **not** store `WANDB_API_KEY` in source, configs, dev_mem, or Git.

## Syncing offline runs

```bash
wandb sync <offline-run-directory>          # one run
wandb sync --sync-all <output_dir>/wandb    # all offline runs in a dir
```

## Comparing Mac and DGX runs

Use a shared `project` and meaningful `group`/`tags` (e.g. `--run-name` or config `group:
25m-portability`). Auto run names encode `<model>-<platform>-<experiment>-<timestamp>`
(e.g. `bert27m-cuda-dgx_portability-20260711-183000`), so the platform is visible in the run
list. Compare `eval/loss`, `eval/perplexity`, `train/tokens_per_second`, and the logged config
(precision, device, SDPA, batch, seq) side by side.

## What is logged

- **Config (run identity):** git commit + dirty, resolved config, param count, architecture,
  seed, dataset/tokenizer identity, batch / grad-accum / effective batch, seq length, optimizer,
  scheduler, precision, device, PyTorch/CUDA version, GPU name, BF16 support, SDPA, checkpoint
  policy.
- **Metrics (with explicit `global_step`):** `train/{loss,learning_rate,gradient_norm,
  tokens_seen,tokens_per_second,steps_per_second}`, `eval/{loss,perplexity,masked_accuracy,
  masked_tokens}`, `system/{peak_ram_mb,peak_vram_mb}`, `analysis/{status,
  predicted_asymptotic_loss,recommended_stop_step,recent_improvement}`.
- **Summary:** final step, final/best val loss + step, final train loss, perplexity, masked
  accuracy, elapsed seconds, total tokens, mean tokens/s, stop reason, checkpoint path,
  analysis status.

## Artifacts

Local checkpoints remain authoritative — **not every checkpoint is uploaded.** Optional,
typed artifacts (only when enabled): `config` (resolved_config.yaml, environment.json),
`metrics` (metrics.jsonl), `report`/plots (from the analyzer), and `model` (the best/final
checkpoint) **only** with `log_checkpoints: true` / `--wandb-log-checkpoints`. Never uploaded:
datasets, raw text, API keys, secrets, or arbitrary local files.

## Privacy & secret handling

- Config/summary payloads are **redacted** of any key that looks like an api key/token/password
  before being sent to W&B.
- No secrets are stored in the repo. Online auth uses the W&B CLI / `WANDB_API_KEY` env var,
  which this project never reads, logs, or commits.
- The `<output_dir>/wandb/` directory is git-ignored.

## Interaction with local JSONL + curve analysis

The `--metrics-file` JSONL and `scripts/analyze_training_curve.py` static figures/reports remain
the project's analysis path. When tracking is active, the analyzer's status/forecast is *also*
logged to the same run as `analysis/*` metrics and (optionally) the report/plots as artifacts —
but the local files are authoritative and the custom analyzer is not replaced by W&B dashboards.

## Troubleshooting

- **"wandb selected but not installed":** `pip install -e ".[wandb]"`.
- **Offline run won't sync:** run `wandb login` first, then `wandb sync <dir>`.
- **Don't want any tracking:** leave `tracking.backend: none` (default) or omit `--wandb`.
- **Online init hangs:** you are probably offline — use `--wandb-mode offline`.
- **Accidentally logged something sensitive:** it is redacted by key name; if you used an
  unusual key name, delete the run in the W&B UI and add the name to `_SECRET_MARKERS`.
