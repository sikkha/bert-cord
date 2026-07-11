# CLAUDE.md — Working notes for AI assistants on `bert_cord`

This file orients any AI assistant (Claude Code / Cowork) working in this repository.
Read `dev_mem/project_brief.md` first — it is the authoritative specification.

## What this project is

`bert_cord` is research toward a **very small BERT-style AI coordination model** (target
ceiling ~200M params). The coordinator learns to interpret system state and decide when to
handle events locally, delegate to a larger LLM, activate memory, request clarification, or
control task lifecycle — *without* doing the full reasoning itself.

**Milestone 0 (current):** only a clean, runnable, configurable **~25M-parameter custom BERT
masked-language-model (MLM) pretraining system**. Nothing else.

## Hard rules

1. **Do not** use Hugging Face `BertModel` or `AutoModelForMaskedLM` internally. The encoder
   is implemented from scratch in `src/coordinator_bert/`.
2. HF `datasets`, `tokenizers`, and `accelerate` **may** be used (data/tokenizer/loop only).
3. Correctness before optimization. Readable research code over framework abstractions.
4. Configuration-driven: all architecture/training knobs live in `configs/*.yaml`.
5. **Never claim a test or smoke run passed without inspecting its actual output.**
6. Distillation and coordination heads stay **separate** from base MLM and are **not**
   implemented in Milestone 0 (placeholders only).
7. Keep `dev_mem/` updated every meaningful session (see policy below).

## Milestone 0 non-goals

Qwen distillation, teacher hidden-state alignment, external LLM routing, coordination heads,
RL, evolutionary optimization, realtime voice, Griffin/recurrent replacement, large-corpus
prep, and full 100M/200M training. Do **not** start these.

## Layout

```
src/coordinator_bert/   # custom model + data + checkpointing (importable package)
scripts/                # train_tokenizer, pretrain_mlm, evaluate (+ placeholders)
configs/                # bert_25m.yaml (real), bert_100m/200m.yaml (provisional)
tests/                  # pytest suite — must pass before any success claim
experiments/smoke/      # tiny smoke-training outputs & checkpoints
dev_mem/                # append-only logs + status + ADRs + experiment log
```

## dev_mem policy (mandatory)

- `development_log.md` — **append-only** chronological record. Never overwrite history.
- `current_status.md` — concise latest verified state; rewrite when status changes.
- `architecture_decisions.md` — ADRs for design choices.
- `experiment_log.md` — every smoke/train run: config, hardware, command, runtime, loss,
  masked accuracy, throughput, memory, checkpoint path, interpretation.

## How to run (Milestone 0)

```bash
pip install -e ".[data,dev]"          # or: pip install torch pyyaml datasets tokenizers accelerate pytest
python -m pytest -q                    # full test suite
python scripts/pretrain_mlm.py --config configs/bert_25m.yaml --smoke   # short synthetic run
python scripts/pretrain_mlm.py --config configs/bert_25m.yaml --smoke --resume experiments/smoke/checkpoints/last
```

The training entrypoint reports OS/arch, Python, PyTorch, CUDA, device, BF16 support, seed,
and parameter count at startup, then trains, evaluates (val loss + masked accuracy), and
checkpoints (model/optimizer/scheduler/step/RNG).
