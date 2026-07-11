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
