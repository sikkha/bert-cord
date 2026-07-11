# Release checklist (pre-GitHub)

Run through this before pushing `bert_cord` to GitHub and before any DGX bring-up.

## Packaging

- [ ] `python -m pip install -e ".[dev,train,analysis]"` succeeds in a clean venv.
- [ ] Core deps are minimal (`torch`, `numpy`, `pyyaml`); everything else is an extra.
- [ ] Optional extras resolve: `dev`, `train`, `analysis`, `scipy_optional`, `wandb`, `all`.
- [ ] scipy is **not** required — curve fitting works on the numpy-only path.
- [ ] W&B is optional and never imported unless installed + requested.
- [ ] `pip install -e ".[dev,train,analysis,wandb]"` succeeds; project still runs without wandb.

## Experiment tracking (optional W&B)

- [ ] `tracking.backend` defaults to `none`; training is byte-identical with/without tracking.
- [ ] Selecting `wandb` while absent raises an actionable error.
- [ ] Offline run needs no auth/network; prints the exact `wandb sync <dir>` command; never
      auto-syncs.
- [ ] Config/summary are redacted of secret-like keys; no `WANDB_API_KEY` in source/config/git.
- [ ] Checkpoint artifacts off by default (`log_checkpoints: false`); local checkpoints remain
      authoritative.
- [ ] `docs/WANDB_INTEGRATION.md` present and linked from README.

## Correctness

- [ ] `python -m pytest -q` — all pass (+1 intentional distillation xfail).
- [ ] Feature-resolution tests pass without real CUDA (`tests/test_runtime.py`).
- [ ] Checkpoint tests pass, incl. checksum verification + corrupted-checkpoint rejection.
- [ ] Curve-analysis + plotting tests pass headless.

## Configuration

- [ ] `configs/model|platform|experiments|examples` present and compose via `extends`.
- [ ] Resolved configs load: `bert_25m_mac`, `bert_25m_dgx_portability`,
      `bert_25m_dgx_throughput`, `bert_100m_dgx`, `bert_200m_dgx`.
- [ ] Portability profile matches Mac math exactly except precision (bf16 vs fp32).
- [ ] Startup prints the fully resolved runtime; disabled features print notes.
- [ ] Scientific settings are never silently altered by platform.

## Diagnostics & benchmarking

- [ ] `check_environment.py` prints text + writes JSON; `--require training` passes on Mac.
- [ ] `check_environment.py --require dgx` exits non-zero off-CUDA (as intended on the Mac).
- [ ] `benchmark_training.py` produces `environment.json`, `resolved_config.yaml`,
      `metrics.jsonl`, `benchmark_summary.json`, `benchmark_report.md`, `plots/`.
- [ ] Benchmark caps at 200 steps; batch probe is opt-in and OOM-safe.

## Checkpoints

- [ ] Immutable `step_XXXXXX/` dirs + `latest.json` pointer; no 300 MB "last" duplicate.
- [ ] Atomic writes; SHA-256 stored in `metadata.json`; verify-on-load works.
- [ ] Resume from checkpoint root follows `latest.json` and restores the global step.

## Git hygiene

- [ ] `.gitignore` excludes `.venv/`, caches, `.DS_Store`, `data/`, `datasets/`, `outputs/`,
      `wandb/`, `experiments/` (except `.gitkeep`), checkpoints, `*.pt|*.pth|*.ckpt|*.safetensors|*.bin`,
      generated `*.png|*.svg|*.pdf` (except `docs/**`), and secrets.
- [ ] `git status`, `git diff --check`, `git ls-files` reviewed — no checkpoints, venv,
      secrets, large binaries, or `.DS_Store` tracked.
- [ ] The tiny analyzer fixture `tests/fixtures/curve_metrics.jsonl` **is** tracked.
- [ ] No destructive history rewrite performed.
- [ ] If `.git/HEAD.lock` exists on the real Mac, remove it **only after** confirming no git
      process is running: `rm -f .git/HEAD.lock`.

## Documentation

- [ ] README has minimal / dev / train / analysis / wandb install and the Mac+DGX pipelines.
- [ ] `docs/DGX_DEPLOYMENT.md` covers install, validation, exact commands, acceptance criteria,
      conservative edit policy, troubleshooting, and reporting-back.
- [ ] `docs/training_curve_analysis.md` states the heuristic limitations.
- [ ] `CLAUDE.md` contains the conservative DGX edit policy.
- [ ] `dev_mem/` updated (development_log, current_status, architecture_decisions, experiment_log).

## Final commit & tag (run on the Mac)

```bash
rm -f .git/HEAD.lock                       # only if present and no git process is running
git add -A
git rm --cached experiments/smoke/smoke_train.log 2>/dev/null || true   # now git-ignored
git status && git diff --check
git commit -m "Release prep: dual-platform packaging, platform configs, diagnostics, \
benchmark, immutable checkpoints, docs"
git tag -a v0.1.0-rc1 -m "bert_cord 25M MLM — dual-platform release candidate (DGX-ready, \
not DGX-validated)"
```

Do **not** push automatically or create the remote automatically. Do **not** claim DGX
compatibility until the actual DGX run passes.
