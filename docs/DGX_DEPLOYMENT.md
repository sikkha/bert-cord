# DGX Spark deployment guide

How to take `bert_cord` from GitHub `main` onto an NVIDIA DGX Spark and run the initial
CUDA/BF16 **portability** and **throughput** experiments. This is platform bring-up, not the
long training run.

> **Status honesty:** as of this release the CUDA/BF16 path is feature-detected and unit-tested
> via mocks but has **not** been executed on real NVIDIA hardware. Treat everything below as
> *readiness*. Do **not** claim DGX compatibility until the actual DGX run passes.

## 1. Installation

```bash
git clone <repo-url> bert_cord && cd bert_cord
git checkout <release-tag-or-commit>          # exact tag/commit — see policy below
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,train,analysis]"
```

If PyTorch's default wheel is not CUDA-enabled on the box, install the CUDA build first
(matching the driver), then `pip install -e ".[dev,train,analysis]"`.

## 2. Environment validation

```bash
python scripts/check_environment.py --require dgx --json experiments/dgx_env.json
```

This **must exit 0**. It fails (non-zero) if CUDA is unavailable, no GPU is visible, the project
dir is not writable, or critical training packages are missing. Add `--require-bf16` to also
require BF16. Inspect the printed report: device `cuda`, a GPU name, compute capability,
`bf16_supported=True`, cuDNN version, SDPA availability.

## 3. Exact command sequence (required order)

```bash
# 1. environment check
python scripts/check_environment.py --require dgx

# 2. tests
python -m pytest -q                      # expect: all pass, 1 xfailed

# 3. portability benchmark (same math as the Mac run; bf16/CUDA, SDPA off)
python scripts/benchmark_training.py \
  --config configs/bert_25m_dgx_portability.yaml \
  --steps 50 --eval --checkpoint \
  --output-dir experiments/benchmarks/dgx_portability

# 4. checkpoint / resume test
python scripts/pretrain_mlm.py --config configs/bert_25m_dgx_portability.yaml \
  --smoke --max-steps 20
python scripts/pretrain_mlm.py --config configs/bert_25m_dgx_portability.yaml \
  --smoke --max-steps 40 --resume experiments/smoke_dgx/checkpoints   # follows latest.json

# 5. throughput benchmark (bf16, SDPA on, larger batch/seq, pinned memory, workers)
python scripts/benchmark_training.py \
  --config configs/bert_25m_dgx_throughput.yaml \
  --steps 200 --output-dir experiments/benchmarks/dgx_throughput

# 6. ONLY THEN longer training (with explicit approval)
```

Optional bounded VRAM headroom probe (diagnostic; never modifies the config):

```bash
python scripts/benchmark_training.py --config configs/bert_25m_dgx_throughput.yaml \
  --steps 20 --probe-batch-sizes --probe-candidates 8 16 32 64 128 \
  --output-dir experiments/benchmarks/dgx_probe
```

## 4. Acceptance criteria

Bring-up is accepted when:

1. `check_environment.py --require dgx` exits 0 and reports `bf16_supported=True`.
2. `pytest -q` passes (all green, 1 intentional xfail).
3. Portability benchmark runs with resolved **device=cuda, precision=bf16, SDPA off**, produces
   `benchmark_summary.json` with finite steps/s and tokens/s, and no NaN/Inf loss.
4. A smoke run checkpoints to an immutable `step_XXXXXX/` dir and **resume restores the global
   step** with checksum verification passing (no loss spike at resume).
5. Throughput benchmark runs with **SDPA on, pinned memory, workers>0**, reports peak VRAM, and
   throughput ≥ the portability profile.
6. Startup logs show the fully resolved runtime, and any disabled feature has a printed note.

Record the resulting `environment.json`, `benchmark_summary.json`, and
`benchmark_report.md` and report them back (see §7).

## 5. Conservative DGX edit policy

- **GitHub `main` is the source of truth.** DGX begins from an exact tag or commit.
- The **first DGX run must use a clean working tree** (`git status` clean).
- **Change config files before Python source.** Prefer config-only changes.
- A DGX experiment may edit **at most one config file** and, **only if unavoidable, one Python
  implementation file**.
- Every DGX **source** edit must: happen **on a branch**, **pass `pytest -q`**, show `git diff`,
  be **committed separately**, and be **pushed for Mac-side review**.
- **No uncontrolled long training.** **No more than 200 benchmark steps** before explicit
  approval (the benchmark script enforces a 200-step cap).
- **No deletion of existing checkpoints by diagnostic scripts.**
- **No `sudo` or system-level changes** from project tooling.

## 6. Troubleshooting

- **`--require dgx` fails on CUDA:** verify the driver + a CUDA-enabled torch
  (`python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`).
- **bf16 unavailable:** older GPUs lack it; the trainer falls back to fp32 and prints a note.
  For portability comparison this is acceptable (fp32), but flag it in the report.
- **OOM in throughput:** lower `per_device_batch_size` / `max_seq_length` in a **copy** of the
  throughput config (never edit the tracked one for a quick test without a branch). Use the
  bounded batch probe to find headroom.
- **Slow dataloader:** increase `runtime.num_workers` (persistent workers require workers>0).
- **torch.compile issues:** it is **off by default**; only enable `runtime.torch_compile: true`
  deliberately, and expect a first-step compile cost.
- **Checksum mismatch on resume:** the checkpoint is corrupted; use an earlier immutable
  `step_XXXXXX/` dir (they are never overwritten).

## 7. Reporting results back to the Mac / GitHub workflow

1. Copy `experiments/**/environment.json`, `benchmark_summary.json`, `benchmark_report.md`, and
   `analysis/` off the DGX (do **not** commit large checkpoints or generated figures — they are
   git-ignored).
2. If any source was changed, push the review branch and open a PR against `main`; include the
   `git diff` and the passing `pytest` output.
3. Append a short entry to `dev_mem/experiment_log.md` (hardware, commit, command, throughput,
   precision, peak VRAM, interpretation) and update `dev_mem/current_status.md`.
4. Only after Mac-side review of the portability numbers should longer training be scheduled.
