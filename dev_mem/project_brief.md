# BERT Coordinator Project Brief

## Project Name

`bert_cord`

## Working Directory

```bash
$HOME/code/bert_cord
```

## Purpose

This project explores a very small AI coordination model that does not attempt to be a general-purpose chatbot or standalone AGI system.

The initial research hypothesis is:

> A compact BERT-style encoder, capped at approximately 200 million parameters, can learn to interpret system state and coordinate larger external AI units without needing to perform the full reasoning task itself.

The small model should eventually decide when to:

- handle a simple event locally,
- delegate to a larger core LLM,
- activate memory or another specialist,
- request clarification,
- continue, pause, interrupt, resume, or terminate a task,
- select what context should be transferred between AI units.

The first implementation stage is deliberately narrower. We will first build and validate a clean, configurable BERT masked-language-model pretraining codebase. Distillation and coordination heads will be added only after the base model and training pipeline are proven correct.

## Research Direction

The project is inspired by several related ideas:

- BERT-style bidirectional semantic encoding,
- compact coordination models such as TRINITY,
- distillation from larger teachers such as Qwen,
- explicit external coordination among AI units,
- possible later comparison with recurrent or Griffin-style architectures.

The BERT implementation serves as the first baseline because it is easier to inspect, train, validate, and compare than a more experimental recurrent architecture.

## Parameter Constraint

The first-stage research ceiling is:

```text
Maximum target size: approximately 200M parameters
```

Development should proceed through smaller configurations first:

```text
Micro model:      approximately 20M-30M
Base experiment:  approximately 90M-120M
Maximum model:    approximately 180M-200M
```

Do not begin by training the largest model.

## Development Principles

1. Correctness before optimization.
2. Readable research code over framework complexity.
3. Configuration-driven architecture.
4. Reproducible experiments.
5. Actual tests and smoke runs before claiming success.
6. No hidden dependence on Hugging Face `BertModel` internals.
7. Hugging Face datasets, tokenizers, and Accelerate may be used.
8. Distillation and coordination logic must remain separate from base MLM pretraining.
9. Development status must be recorded in Markdown inside `dev_mem/`.
10. Any incomplete, failed, or uncertain result must be documented honestly.

## Initial Repository Structure

```text
bert_cord/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── bert_25m.yaml
│   ├── bert_100m.yaml
│   └── bert_200m.yaml
├── dev_mem/
│   ├── project_brief.md
│   ├── development_log.md
│   ├── current_status.md
│   ├── architecture_decisions.md
│   └── experiment_log.md
├── src/
│   └── coordinator_bert/
│       ├── __init__.py
│       ├── configuration.py
│       ├── embeddings.py
│       ├── attention.py
│       ├── model.py
│       ├── mlm_head.py
│       ├── masking.py
│       ├── data.py
│       ├── checkpointing.py
│       ├── distillation.py
│       └── coordination_heads.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── pretrain_mlm.py
│   ├── distill_teacher.py
│   ├── train_coordinator.py
│   └── evaluate.py
├── tests/
│   ├── test_attention.py
│   ├── test_masking.py
│   ├── test_model_shapes.py
│   ├── test_checkpoint.py
│   └── test_distillation.py
└── experiments/
    └── smoke/
```

Files related to distillation and coordination may initially contain documented placeholders, but they should not be implemented in Milestone 0.

## `dev_mem` Policy

The `dev_mem/` directory is mandatory.

Claude Code or Claude Cowork must update it during every meaningful development session.

### Required files

#### `development_log.md`

Append-only chronological record containing:

- date and time,
- task attempted,
- files changed,
- commands run,
- tests executed,
- results,
- failures,
- unresolved questions,
- next recommended step.

#### `current_status.md`

A concise current checkpoint containing:

- what currently works,
- what is partially working,
- what is broken,
- latest verified command,
- latest verified checkpoint,
- immediate next task.

This file should be rewritten when project status changes.

#### `architecture_decisions.md`

Record important design decisions using this form:

```markdown
## ADR-001: Decision title

Date:

Status: proposed / accepted / superseded

Context:

Decision:

Alternatives considered:

Consequences:
```

#### `experiment_log.md`

Record all smoke tests and training experiments:

- configuration file,
- git commit if available,
- hardware,
- dataset,
- command,
- runtime,
- loss,
- masked-token accuracy,
- throughput,
- memory use,
- checkpoint path,
- interpretation.

No experiment should be described as successful unless its actual output was inspected.

## Milestone 0

### Goal

Create a runnable, testable, configurable 25M-parameter BERT masked-language-model pretraining system.

### Required capabilities

- custom PyTorch BERT implementation,
- bidirectional multi-head self-attention,
- learned token embeddings,
- learned position embeddings,
- token-type embeddings if retained,
- configurable pre-layer normalization or post-layer normalization,
- GELU feed-forward network,
- residual connections,
- tied input and output token embeddings,
- dynamic MLM masking,
- standard 15% token selection,
- standard 80/10/10 replacement behavior,
- attention masks,
- BF16 support where available,
- Hugging Face Accelerate integration,
- gradient accumulation,
- AdamW optimizer,
- warmup plus decay scheduler,
- deterministic random seeds,
- YAML configuration,
- checkpoint save,
- checkpoint resume,
- validation loss,
- masked-token accuracy,
- parameter-count report,
- synthetic or tiny-dataset smoke test,
- unit tests.

### Non-goals for Milestone 0

Do not yet implement:

- Qwen distillation,
- teacher hidden-state alignment,
- external LLM routing,
- coordination heads,
- reinforcement learning,
- evolutionary optimization,
- realtime voice integration,
- Griffin or recurrent replacement,
- large-scale corpus preparation,
- 100M or 200M full training.

## Model Configuration

The code must calculate actual parameter counts.

The configuration names are targets, not assumptions.

Example model tiers:

```yaml
# configs/bert_25m.yaml
model:
  vocab_size: 32000
  hidden_size: 384
  num_hidden_layers: 8
  num_attention_heads: 6
  intermediate_size: 1536
  max_position_embeddings: 512
  type_vocab_size: 2
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
  layer_norm_eps: 1.0e-12
  norm_type: post
```

Claude Code may revise these dimensions to reach the approximate target, but must document the actual parameter count.

## Implementation Guidance

The model should remain easy to inspect.

Recommended module boundaries:

```text
configuration.py
    dataclass or validated config structures

embeddings.py
    token, position, and token-type embeddings

attention.py
    multi-head bidirectional self-attention

model.py
    transformer blocks, encoder stack, pooled outputs

mlm_head.py
    transform, normalization, decoder, tied weights

masking.py
    dynamic MLM corruption

data.py
    dataset preparation and dataloaders

checkpointing.py
    save, resume, metadata, RNG state
```

The implementation may use PyTorch scaled-dot-product attention if behavior is tested and documented.

## Reference Code Policy

External repositories may be inspected for ideas, but copied code must be attributed and licensing must be checked.

Useful references:

- `barneyhill/minBERT`
- Hugging Face Transformers `run_mlm_no_trainer.py`
- `google-research/bert`
- NVIDIA Megatron-LM, later only

Do not clone a large framework into the project unless explicitly approved.

## Testing Requirements

Minimum tests:

### Attention

- output shapes,
- attention-mask behavior,
- bidirectional access,
- no NaNs,
- deterministic behavior with dropout disabled.

### Masking

- approximately 15% selected-token rate over a sufficiently large sample,
- selected tokens follow approximately 80/10/10 behavior,
- special tokens are never masked,
- labels are `-100` for non-selected tokens.

### Model

- forward pass shapes,
- MLM logits shape,
- tied embeddings,
- gradient propagation,
- parameter count output.

### Checkpointing

- save and reload,
- resumed global step,
- optimizer and scheduler restoration,
- reproducible continuation where practical.

### Smoke Training

A short run must demonstrate:

- decreasing or stable loss,
- valid masked-token accuracy,
- checkpoint creation,
- successful resume,
- no runtime error,
- no unsupported precision claim.

## Hardware Context

The intended training hardware is an NVIDIA DGX Spark-compatible system.

The implementation should:

- detect CUDA,
- support BF16 only when hardware and PyTorch report support,
- fall back safely when unavailable,
- report device and precision mode at startup,
- avoid assuming x86-only behavior,
- avoid unnecessary model-parallel complexity.

## Reproducibility

Every training run should record:

```text
seed
configuration
package versions
PyTorch version
CUDA version
device
precision
dataset identity
tokenizer identity
git commit if available
```

## Future Milestones

### Milestone 1: 100M baseline

Scale the verified architecture and training loop.

### Milestone 2: Teacher distillation

Use Qwen or another larger model to teach coordination-relevant representations and structured decisions.

Possible losses:

```text
masked-language modeling
representation alignment
task-type classification
organ selection
control-state prediction
continuation or termination
```

### Milestone 3: Coordination heads

Add factorized heads for:

```text
target organ
operation
priority
control transition
context-transfer policy
continue / pause / interrupt / stop
```

### Milestone 4: External LLM integration

Connect the student coordinator to one or more external core LLMs through a deterministic orchestrator.

### Milestone 5: Architecture comparison

Compare:

```text
BERT encoder
decoder-only transformer
Griffin-style recurrent-attention model
```

## Definition of Success for the First Stage

The first stage succeeds when:

1. the custom 25M BERT implementation passes all tests,
2. MLM smoke training runs on the target machine,
3. loss and masked-token accuracy are reported,
4. checkpoint resume is verified,
5. all commands and results are recorded in `dev_mem/`,
6. the codebase is clean enough to extend toward 100M and distillation.

The project must not claim success based only on code generation or static inspection.
