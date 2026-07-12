# Architecture Decisions

Design decisions for `bert_cord`, recorded as ADRs.

---

## ADR-010: Platform-aware config composition + separate runtime layer

Date: 2026-07-11

Status: accepted

Context: The repo must run on Apple Silicon (MPS) locally and on an NVIDIA DGX Spark (CUDA),
and support exact-portability vs throughput profiles — without silently changing the science.

Decision: Keep the flat-config loader but add an `extends:` composition (deep merge, current
file wins) and a dedicated `runtime` config section (device + perf toggles), kept **separate**
from `model`/`train`. Provide `configs/model/`, `configs/platform/`, `configs/experiments/`,
`configs/examples/`, and self-contained resolved `configs/bert_25m_{mac,dgx_portability,
dgx_throughput}.yaml` (+ 100m/200m dgx). The portability profile matches the Mac math exactly
(same seed/batch/seq/optimizer/scheduler, SDPA off) and differs only in device + precision.
Hardware resolution + feature detection live in `coordinator_bert/runtime.py`; the fully
resolved runtime is printed at startup and written to `resolved_config.yaml`/`environment.json`.

Alternatives: full restructure into a multi-file loader (rejected: more loader risk — the
`extends` overlay achieves the same with backward compatibility); precision in a platform-only
file (kept in `train` since the trainer already resolves it against hardware).

Consequences: flat configs still load unchanged; 12 feature-resolution tests cover CUDA/MPS
behaviour without real hardware. Scientific settings never change silently by platform.

## ADR-011: Immutable, checksum-verified checkpoints

Date: 2026-07-11

Status: accepted

Context: The previous overwrite-heavy `last/` behaviour was ambiguous and, on a synced mount,
did not reliably persist overwrites of the 300+ MB state file.

Decision: `CheckpointManager` writes **immutable** `step_XXXXXX/` directories via atomic
rename (temp dir → `os.replace`), a separate `metadata.json` (global step, git commit, config
hash, param count, precision, device, SHA-256, timestamp, torch version), and a tiny
`latest.json` pointer that also records `best` (a pointer, never a duplicated copy). Loads can
verify the SHA-256 and reject corrupted checkpoints. Legacy `save_checkpoint`/`load_checkpoint`
are preserved (now atomic + checksummed). Resume accepts a step dir, a `state.pt`, or a
checkpoint root (resolved via `latest.json`).

Alternatives: keep overwrite `last` (rejected: ambiguous, unreliable); duplicate best copy
(rejected: wastes hundreds of MB — a pointer suffices).

Consequences: no 300 MB duplication; 9 checkpoint-manager tests cover atomic save, load-by-path,
latest/best pointers, checksum pass/fail, corruption rejection, and resume-step restoration.

## ADR-012: Packaging extras (minimal core; opt-in train/analysis/wandb)

Date: 2026-07-11

Status: accepted

Context: Core install should only need what runs the model; heavy/opt deps should be extras.

Decision: Core = `torch, numpy, pyyaml`. Extras: `dev` (pytest), `train` (accelerate, datasets,
tokenizers, safetensors, psutil), `analysis` (matplotlib), `scipy_optional`, `wandb`, and
`all` (= dev+train+analysis). scipy stays optional — the numpy-only curve fitting is the
default path and is what CI exercises. W&B is never mandatory.

Consequences: `pip install -e ".[dev,train,analysis]"` is the standard dev/DGX install; a bare
`pip install -e .` imports and runs the model. Verified in the active environment.

## ADR-013: Optional W&B tracking behind a backend-agnostic interface

Date: 2026-07-11

Status: accepted

Context: We want optional experiment tracking for Mac↔DGX comparison without making W&B
mandatory, changing training math, or weakening the local JSONL/curve-analysis pipeline.

Decision: Add `coordinator_bert/tracking.py` with a tiny interface (`init_run`, `log_metrics`,
`log_summary`, `log_file`, `log_artifact`, `finish`) and two backends: `NullTracker` (default,
no-op) and `WandbTracker` (lazy `import wandb` — never at module load). A `TrackingConfig`
(default `backend: none`) drives it; the trainer logs consistently-named metrics
(`train/*`, `eval/*`, `system/*`, `analysis/*`) with explicit `global_step`, a run summary, and
optional typed artifacts (`config`/`metrics`/`report`/`model`). `finish()` is called in a
`finally` block so it always runs (incl. exceptions). Config/summary are redacted of
secret-like keys; offline mode needs no auth/network and prints an explicit `wandb sync`
command (never auto-syncs). DGX profiles default to `wandb`/`offline`; Mac defaults to `none`.

Alternatives: hard W&B dependency (rejected: must stay optional); replacing the custom analyzer
with W&B dashboards (rejected: local JSONL + curve analysis remain the source of truth);
uploading every checkpoint (rejected: local checkpoints authoritative; `log_checkpoints` off by
default).

Consequences: 17 tracking tests (mocked wandb — no network) cover null no-op, absent-package
error, offline init, metric names/step, finish-on-exception, secret redaction, and
artifacts-off-by-default. Verified with a live offline run (no network). Training is byte-
identical with tracking off.

## ADR-014: ONNX export + ONNX Runtime for portable MLM inference (Milestone 0.7)

Date: 2026-07-11

Status: accepted

Context: We want portable, framework-neutral inference for the encoder-only MLM without the
training stack, and a path toward GPU/other backends later — without changing the architecture
or weakening the authoritative PyTorch checkpoints.

Decision:
- **Format = ONNX; runtime = ONNX Runtime.** The model is an encoder-only MLM (single forward,
  per-position logits) — not autoregressive — so **vLLM / Ollama / llama.cpp** (decoder-only,
  KV-cache, GGUF, chat serving) do not apply. ONNX Runtime runs the exact graph on CPU/GPU with
  minimal deps and verifiable numerical parity.
- **I/O contract:** inputs `input_ids`, `attention_mask`, `token_type_ids` (int64
  `[batch, sequence]`); single output `logits` (float32 `[batch, sequence, vocab]`). A thin
  `MLMInferenceWrapper` calls the model with `labels=None`, `return_probs=False` and returns the
  logits tensor — no loss branch, no dict, no variable-length attention-prob output. Architecture
  unchanged.
- **Dynamic shapes:** `batch` and `sequence` are dynamic axes; `vocab` fixed. Verified by running
  ORT at shapes different from the trace.
- **Opset 18:** torch 2.13's exporter implements opset 18 natively; onnx≥1.16 / onnxruntime≥1.17
  support it; lower opsets force a lossy down-conversion. torch 2.x routes `torch.onnx.export`
  through the dynamo exporter (`torch.export.export`) even with `dynamic_axes`; that path produces
  a checker-valid, ORT-executable graph here.
- **Dependency strategy:** `onnx`, `onnxruntime`, `onnxscript` live in an optional `onnx` extra
  (also folded into `all`); never core. Imported lazily with an actionable error if missing.
- **Checkpoints stay authoritative:** ONNX is a derived inference artifact. Optimizer/scheduler/
  RNG/loss/labels/checkpoint-manager logic are **not** exported. Artifacts (`exports/`, `*.onnx`,
  `*.onnx.data`) are git-ignored; distribute via GitHub Releases / Hugging Face.

Alternatives: TorchScript (rejected: still torch-bound, less portable); GGUF/llama.cpp (rejected:
decoder-only LLM tooling, wrong model class); committing the artifact to Git (rejected: ~100 MB).

Consequences: 17 ONNX tests (tiny model, ORT-dependent ones skip if packages absent) plus an
inspected real 27.01M export + parity run (`max|Δ|≈7e-6`, top-5 agreement 1.00, dynamic axes).
FP32 CPU only is validated; CoreML and `onnxruntime-gpu` remain untested (documented honestly).
Large models externalize weights to a sibling `.onnx.data` file that must ship with the graph.

## ADR-015: Hugging Face model-repo packaging for the ONNX baseline (local staging only)

Date: 2026-07-11

Status: accepted

Context: We want a Hugging Face model repository (`sikkha/bert-cord-27m-mlm-onnx`) for the ONNX
inference baseline, prepared locally so it can be uploaded manually later — without uploading,
authenticating, creating the remote, or touching the authoritative PyTorch checkpoints.

Decision:
- **Local staging dir** `bert-cord-27m-mlm-onnx/` (git-ignored), built by
  `scripts/build_hf_onnx_package.py` and checked by `scripts/validate_hf_onnx_package.py`.
  Neither script uploads, authenticates, or accesses the network.
- **External-data relink (critical):** the source weights file is `<model>.onnx.data`, but the
  packaged graph must reference `model.onnx.data`. Renaming is insufficient — the location is
  stored inside the graph — so the builder **re-saves** the model with
  `onnx.save_model(..., save_as_external_data=True, location="model.onnx.data")`, then validates
  the packaged copy from inside the package dir.
- **Package contract:** `README.md` (model card, `library_name: onnxruntime`, no
  `pipeline_tag: fill-mask` — no tokenizer bundled), `LICENSE` (copied), `config.json`
  (architecture + provenance: source commit/tag, params 27,010,304, opset 18, precision fp32),
  `evaluation.json` (freshly measured parity + file checksums + limitations), `requirements.txt`
  (numpy + onnxruntime only), `inference.py` (standalone, no bert_cord dependency), `MANIFEST.json`
  (per-file path/size/SHA-256, excluding itself), `onnx/model.onnx` + `onnx/model.onnx.data`.
- **Honesty constraints baked into the model card:** synthetic MLM baseline, not a coordinator/
  mini-amygdala, no tokenizer, no `AutoModel` compatibility, no language-understanding or
  production-readiness claim; CPU/FP32 validated, CUDA/CoreML/FP16/BF16 unvalidated; both ONNX
  files required together.
- **Four artifact planes kept distinct:** GitHub source (code) · W&B (run history) · Hugging Face
  (portable inference artifact) · DGX PyTorch checkpoints (training source of truth).

Alternatives: `pipeline_tag: fill-mask` / bundling a tokenizer (rejected: no usable NL tokenizer
exists — synthetic ids only); committing the ~100 MB package to Git (rejected: git-ignored;
distribute via HF); auto-upload (rejected: manual, explicit, no auth in tooling).

Consequences: 14 package tests (tiny fixtures, offline) cover build, relink, checksums, JSON
validity, independent inference, missing-`.onnx.data` failure, and validator rejection of
forbidden files / checksum mismatch / absolute-path & secret leaks. The real package builds and
validates 17/17 offline. Upload remains a documented manual step.

## ADR-016: Config-driven tokenizer pipeline (Tokenizer Milestone)

Date: 2026-07-13

Status: accepted

Context: BERT-Cord needs a reproducible tokenizer-training pipeline. The tokenizer will later be
**frozen** and reused for all MLM pretraining, so the pipeline (not immediate quality) is the
priority. The existing `scripts/train_tokenizer.py` was WordPiece-only and not config-driven.

Decision:
- **Corpus prep** in `src/coordinator_bert/corpus.py` (+ `scripts/prepare_tokenizer_corpus.py`):
  reads txt/md/jsonl and optional streamed HF datasets; Unicode-normalizes, drops empties,
  exact-dedups (SHA-1), deterministically shuffles (seeded), computes per-script language stats,
  and writes sharded output + `corpus_manifest.json` (per-file SHA-256) + `corpus_report.md`.
- **Config-driven trainer** in `src/coordinator_bert/tokenizer_train.py` (extends the CLI, keeps
  a legacy single-file WordPiece mode) supporting three algorithms — **byte-level BPE, Unigram
  (+byte fallback, Metaspace), WordPiece** — selected via `configs/tokenizer/*.yaml`
  (vocab_size, normalization, lowercase, byte_fallback, special tokens). Special tokens are
  **pinned to ids 0–4** ([PAD],[CLS],[SEP],[MASK],[UNK]) to match
  `coordinator_bert.data.SpecialTokens`; training **fails** if that integrity check fails. Output
  artifact dir has tokenizer.json, tokenizer_config.json, special_tokens_map.json,
  tokenizer_manifest.json, README.md.
- **Evaluation** in `src/coordinator_bert/tokenizer_eval.py` (+ `scripts/evaluate_tokenizer.py`):
  unknown-token rate, tokens/sentence, tokens/word, round-trip (exact + whitespace-normalized),
  vocabulary utilization, reserved-token integrity → evaluation.json + evaluation.md.
- **Corpus size policy:** real multilingual corpora (EN/TH Wikipedia, OSCAR, mC4, FineWeb, CC100)
  exceed 1 GB, so **nothing large was downloaded**; `docs/recommended_corpus.md` lists ids,
  sizes, subsets, and exact DGX download commands. A ~2.6 KB local sample corpus (git-ignored)
  makes the pipeline runnable/testable offline.
- **No AutoModel / language-understanding claim.** Tokenizer artifacts and corpus outputs are
  git-ignored (`data/`, `artifacts/`); the frozen tokenizer will be a release artifact.

Alternatives: WordPiece-only (rejected: byte-BPE avoids UNK, Unigram suits Thai); committing
corpus/tokenizer artifacts (rejected: large + reproducible from config); auto-downloading
Wikipedia/OSCAR (rejected: >1 GB, per policy).

Consequences: 9 tokenizer tests (tiny fixtures) cover language detection, dedup, deterministic
shuffle, train/load/round-trip/specials for all three algorithms, YAML config, and evaluation
metrics. Full suite 148 passed, 1 xfailed. The three algorithms remain candidates; one will be
selected and frozen after evaluation on the real corpus.

## ADR-009: Conservative, non-learned training-curve analysis + optional early stop

Date: 2026-07-11

Status: accepted

Context: We want to estimate whether a run is still improving enough to justify more compute,
and produce reproducible research figures — without over-claiming. AdamW gives no principled
stopping point, and learned forecasters would be overkill and opaque at this stage.

Decision: Add `src/coordinator_bert/curve_analysis.py` (analysis → structured data only) and
`src/coordinator_bert/curve_plots.py` (matplotlib `Agg`, static figures), coordinated by
`scripts/analyze_training_curve.py`. Analysis: EMA smoothing, step/log-step slopes, plateau
(patience/min_delta/min_evals) and instability (NaN/Inf, robust loss-spike, gradient
excursion) detection, three closed-form fits (`L_inf+A t^-α`, `L_inf+A e^-kt`, `a+b/√t`) via
bounded grid + least squares (scipy optional, numpy default), model choice by held-out tail
RMSE (AIC fallback), bootstrap CIs, and a status in
{CONTINUE, PLATEAU, UNSTABLE, INSUFFICIENT_DATA}. A **probability-like** target heuristic is
explicitly labelled as not calibrated. No ML forecaster is used.

Trainer integration is opt-in: `--metrics-file` logs per-eval JSONL/CSV; `--early-stop-policy
off|warn|stop` defaults to **off**. `stop` may terminate only after min-evals + patience +
**no instability** + predicted-gain-below-threshold, and always saves a final checkpoint.

Alternatives considered: a learned/GP forecaster (rejected: opaque, over-engineered, easy to
over-trust); auto-stop by default (rejected: unsafe — AdamW does not know the optimum);
seaborn/interactive dashboards (rejected: keep v1 static and reproducible).

Consequences: Separation of analysis vs plotting vs CLI keeps everything testable (30 new
tests). Forecasts are honest but limited — unreliable after phase transitions/instability, and
task metrics must be read alongside perplexity. Documented in `docs/training_curve_analysis.md`.

---

## ADR-001: Custom BERT encoder (no Hugging Face `BertModel`)

Date: 2026-07-11

Status: accepted

Context: The project needs an encoder that is easy to inspect and extend toward coordination
heads and distillation. Hard project rule: no hidden dependence on HF `BertModel` /
`AutoModelForMaskedLM` internals.

Decision: Implement embeddings, multi-head self-attention, transformer blocks, encoder stack,
pooler, and MLM head from scratch in `src/coordinator_bert/`. Use PyTorch only. HF
`datasets`/`tokenizers`/`accelerate` are used solely for data, tokenization, and the training
loop's device/precision/accumulation plumbing — never for the model.

Alternatives considered: subclass HF `BertModel` (rejected: violates project rule, hides
mechanics); use nanoGPT-style decoder (rejected: not bidirectional, wrong objective).

Consequences: Full control and transparency; slightly more code to maintain and test. Tests
assert bidirectionality, mask behavior, tied weights, and parameter counts to lock behavior.

---

## ADR-002: 25M configuration and actual parameter count

Date: 2026-07-11

Status: accepted

Context: The brief targets an "approximately 25M" micro model (20M–30M band) with the example
config: vocab 32000, hidden 384, 8 layers, 6 heads, intermediate 1536, max_pos 512,
type_vocab 2, post-LN, tied embeddings.

Decision: Keep the example dimensions unchanged. The **actual parameter count is
27,010,304 (~27.01M)**, computed both by a closed-form estimator
(`ModelConfig.estimate_num_parameters`) and by direct counting
(`count_parameters`) — the two agree exactly. This sits squarely inside the 20–30M micro
band, so no dimension change is warranted.

Note: the token-embedding matrix (vocab × hidden = 32000 × 384 = 12,288,000) is ~45% of all
parameters and is tied to the output decoder, so it is counted once. "25M" is a label, not a
constraint; 27.01M is the honest figure.

Alternatives considered: shrink vocab to ~28k or hidden to 360 to hit 25.0M exactly (rejected:
32000 is a conventional vocab size and the deviation is within the stated approximate band);
untie embeddings (rejected: adds 12.3M params and is contrary to the spec).

Consequences: Config names (`bert_25m`) are approximate. `test_parameter_count_matches_estimate`
guards that the estimator stays exact; startup always prints the real count.

---

## ADR-003: Configurable pre-LN vs post-LN

Date: 2026-07-11

Status: accepted

Context: BERT-original uses post-layer-norm; pre-LN is often more stable for deeper/larger
models and matters for the later 100M/200M tiers.

Decision: Support both via `model.norm_type` ("post" | "pre"). Post-LN: `x = LN(x + Sublayer(x))`.
Pre-LN: `x = x + Sublayer(LN(x))`, with an added final encoder LayerNorm so outputs are
normalized. Default is post (matches the brief's example). The parameter estimator accounts
for the extra final-LN in pre mode.

Alternatives considered: post-LN only (rejected: less flexible for scale-up).

Consequences: One extra LayerNorm in pre mode; both paths are unit-tested for finite output
and correct shapes.

---

## ADR-004: Explicit softmax attention default; SDPA optional

Date: 2026-07-11

Status: accepted

Context: `torch.nn.functional.scaled_dot_product_attention` (SDPA) is faster but hides the
attention matrix, which we want to inspect and test (mask behavior, row sums).

Decision: Default `use_sdpa=false` (explicit Q·Kᵀ/√d + additive mask + softmax). SDPA is
available via config for speed once behavior is trusted. The additive mask sets padded keys
to `finfo.min` so their softmax weight vanishes; a test asserts padded columns get exactly
zero probability.

Alternatives considered: SDPA-only (rejected: cannot return probabilities for tests).

Consequences: The manual path is the tested reference; the SDPA path is shape/finiteness
tested but does not expose probabilities.

---

## ADR-005: Dynamic MLM masking in the collator

Date: 2026-07-11

Status: accepted

Context: Masking should be dynamic (fresh each batch), follow 15% selection and 80/10/10
replacement, never touch special tokens, and use −100 for unselected labels.

Decision: Implement masking in `MLMasker` and apply it inside `MLMCollator` at batch draw
time (not baked into the dataset). Padding is treated as special (never masked). A per-loader
`torch.Generator` seed makes eval masking reproducible while training masking still varies.

Alternatives considered: static pre-masking (rejected: reduces effective data diversity).

Consequences: Statistics are verified over a large sample (selection ~15%, split ~80/10/10);
special-token safety and −100 labeling are asserted.

---

## ADR-006: BF16 only when hardware reports it; fp32 fallback

Date: 2026-07-11

Status: accepted

Context: Target hardware is an NVIDIA DGX Spark (CUDA + BF16). The current dev/CI machine is
CPU-only aarch64 with no CUDA.

Decision: `resolve_precision` returns bf16 only when `torch.cuda.is_available()` **and**
`torch.cuda.is_bf16_supported()`. "auto" → bf16 on capable CUDA, else fp32. Requested bf16/fp16
without support prints a fallback notice and uses fp32. Accelerate's `mixed_precision` is set
from the resolved value. The GPU BF16 path is therefore **untested on real hardware in this
session** and is exercised only on the DGX Spark.

Alternatives considered: force bf16 (rejected: unsafe, false precision claims); autocast on CPU
bf16 (rejected: slow and not representative).

Consequences: No unsupported-precision claims. Startup prints the resolved device/precision.
Smoke runs here are fp32 on CPU.

---

## ADR-007: Checkpoint = full training state, single write per save

Date: 2026-07-11

Status: accepted

Context: Resume must restore model, optimizer, scheduler, global step, and RNG. The dev disk
is small (~14 GB free) and slow; the 27M model checkpoint is ~323 MB (weights + AdamW moments).

Decision: `save_checkpoint` writes one `state.pt` (all tensors) plus a small readable
`meta.json`. Periodic training saves may mirror to `last`; the smoke profile disables periodic
saves and writes a single `last` at the end to bound I/O. RNG state for Python/NumPy/Torch
(and CUDA when present) is saved and restored.

Alternatives considered: HF Accelerate `save_state` (rejected: opaque sharding, less control);
weights-only checkpoints (rejected: cannot resume optimizer/scheduler).

Consequences: A resume test asserts A→save→resume→B equals uninterrupted A+B bit-for-bit on
CPU. Tied embeddings load correctly because the decoder weight is re-tied at model construction.

---

## ADR-008: Learnable synthetic corpus for smoke training

Date: 2026-07-11

Status: accepted

Context: A smoke run must show *decreasing or stable loss* and a *valid* masked-token accuracy,
proving the gradient/optimizer path actually works — not just that code runs. Purely random
token sequences have no learnable structure, so loss stays flat and accuracy stays ~0.

Decision: The synthetic generator tiles a short random motif (period 3, drawn from a small
64-token sub-vocabulary) to fill each sequence. A masked token can be recovered by copying the
token `period` positions away, giving the MLM real bidirectional signal. Train and val use
different seeds, so falling val loss reflects learning the copy *rule*, not memorization.

Alternatives considered: random tokens (rejected: flat loss, uninformative smoke);
downloading a real corpus (rejected: offline CI, Milestone-0 non-goal of large-corpus prep).

Consequences: Smoke loss falls cleanly (10.4 → 6.1 over 40 steps across a run + resume).
Masked top-1 accuracy stays low/coarse (~0–2%) because eval holds only ~100 masked tokens and
top-1 over the full 32000-vocab is hard this early — documented honestly in the experiment log.
