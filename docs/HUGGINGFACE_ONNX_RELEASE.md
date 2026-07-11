# Hugging Face ONNX release (BERT-Cord 27M MLM baseline)

How to build, validate, and later **manually** publish the ONNX inference package to Hugging
Face. This repo never uploads, authenticates, or creates the remote — those are manual steps you
run yourself.

## At a glance

- **Local staging directory:** `bert-cord-27m-mlm-onnx/` (git-ignored; large artifacts).
- **Future HF repo id:** `sikkha/bert-cord-27m-mlm-onnx` (model type).
- **Package version:** `0.1.2-hf-onnx`.
- **Purpose:** a portable, framework-neutral ONNX Runtime inference artifact for the custom
  27,010,304-param `BertForMaskedLM` — **masked-token prediction only**, FP32, opset 18.
- **Not** the mini-amygdala coordinator: this is a **synthetic MLM development baseline** with no
  tokenizer, no language understanding, and no coordination behavior.

## Package layout

```
bert-cord-27m-mlm-onnx/
├── README.md          model card (YAML front matter; library_name: onnxruntime)
├── LICENSE            Apache-2.0 (copied from the source repo)
├── config.json        architecture + provenance (model/packaging commit+tag, params, opset)
├── evaluation.json    measured parity + file checksums + limitations
├── requirements.txt   numpy>=1.24, onnxruntime>=1.17  (inference-only)
├── inference.py       standalone ONNX Runtime example (no bert_cord dependency)
├── MANIFEST.json      every file: relative path, byte size, SHA-256 (excludes itself)
└── onnx/
    ├── model.onnx       graph (references model.onnx.data)
    └── model.onnx.data  external FP32 weights (~102 MB) — ship together
```

## Build

```bash
python scripts/build_hf_onnx_package.py \
  --config configs/bert_25m_mac.yaml \
  --onnx-model exports/bert_cord_27m_mlm.onnx \
  --output bert-cord-27m-mlm-onnx \
  --repo-id sikkha/bert-cord-27m-mlm-onnx \
  --package-version 0.1.2-hf-onnx
```

The builder detects the source Git commit/tag, safely recreates the output dir, **re-saves the
ONNX so the graph references `model.onnx.data`** (renaming alone is not enough — the external
location is stored inside the graph), re-measures PyTorch↔ONNX parity, generates all metadata
(config/eval/README/inference/requirements/LICENSE/MANIFEST), validates the packaged copy, and
prints the later upload commands. It never uploads.

## Validate

```bash
python scripts/validate_hf_onnx_package.py bert-cord-27m-mlm-onnx
```

Offline; checks required/forbidden files, JSON validity, README front matter, that
`packaging_source_commit` matches the current Git HEAD (model-source provenance is tracked
separately and need not equal HEAD), MANIFEST checksums, `onnx.checker`, ONNX Runtime load +
I/O contract, dynamic batch/sequence inference, `inference.py` execution, and the absence of
leaked absolute paths / secrets. No network.

## Manual inspection checklist

- [ ] `README.md` model card is honest (synthetic MLM baseline; no tokenizer; CPU/FP32 only).
- [ ] `config.json` params = 27,010,304; opset 18; `model_source_*` and `packaging_source_*`
      commit/tag are correct and distinct where applicable.
- [ ] `evaluation.json` parity: `max|Δ|` ≪ 2e-3, top-5 agreement 1.00, no NaN/Inf.
- [ ] `onnx/model.onnx` + `onnx/model.onnx.data` both present; graph references `model.onnx.data`.
- [ ] `python bert-cord-27m-mlm-onnx/inference.py` runs and prints logits shape + top-5.
- [ ] No private absolute paths / secrets anywhere in the package.

## Later: publish to Hugging Face (manual — you run these)

Install the CLI (once): `pip install -U "huggingface_hub[cli]"`.

1. **Login** (stores your token via the HF CLI; never in this repo):
   ```bash
   hf auth login          # or: huggingface-cli login
   ```
2. **Create the remote repo** (model type):
   ```bash
   hf repo create sikkha/bert-cord-27m-mlm-onnx --type model
   ```
3. **Upload the whole package directory:**
   ```bash
   hf upload sikkha/bert-cord-27m-mlm-onnx bert-cord-27m-mlm-onnx .
   ```
   (Uploads all files including `onnx/model.onnx.data`. Large-file handling is automatic.)

## Post-upload validation

- Open `https://huggingface.co/sikkha/bert-cord-27m-mlm-onnx`; confirm `model.onnx` **and**
  `model.onnx.data` are both present, and the model card renders.
- Download both ONNX files and re-run `inference.py` from the downloaded copy.
- Confirm `config.json` / `evaluation.json` / `MANIFEST.json` are intact.

## Update / versioning procedure

Re-export the ONNX from the current checkpoint, bump `--package-version` (e.g. `0.1.3-hf-onnx`),
rebuild, re-validate, then re-upload. Metadata pins provenance via two separate pairs:
`model_source_commit`/`model_source_tag` (the commit the ONNX was exported from) and
`packaging_source_commit`/`packaging_source_tag` (the tooling commit that built the package).
The builder aborts (non-zero exit) rather than reuse a directory it cannot cleanly remove, so
each build is a fresh, consistent package — pass `--model-source-commit`/`--model-source-tag`
when the export commit differs from the packaging commit.

## External-data requirement

`model.onnx` stores weights **externally** in `model.onnx.data`. Both files must be downloaded
together and kept side by side; loading the graph without the data file fails.

## Privacy & security checks

The validator scans all text/JSON for absolute-path leaks (`/Users/`, `/home/…`, `/sessions/`,
Windows drive paths) and secret markers (API-key/token/password patterns, AWS keys, private-key
blocks). No `WANDB_API_KEY` or HF token is stored in the repo, package, or metadata. Online HF
auth uses the HF CLI's own token store, not this project.

## The four artifact planes (do not conflate)

| plane | where | authoritative for | contains |
|-------|-------|-------------------|----------|
| **GitHub source** | github.com/sikkha/bert-cord | the code | src/, scripts/, configs/, docs/, tests/ |
| **W&B tracking** | wandb (optional, offline default) | run history/metrics | logged metrics/summaries (no weights) |
| **Hugging Face artifacts** | HF model repo (this package) | portable inference | ONNX graph + weights + metadata |
| **DGX checkpoints** | DGX filesystem | **training source of truth** | PyTorch model+optimizer+scheduler+RNG |

The PyTorch checkpoints remain authoritative; the HF ONNX package is a **derived inference
artifact**.

## Why this is a baseline, not the mini-amygdala release

This packages the current Milestone-0.7 synthetic MLM encoder purely so a portable inference
artifact exists and the release pipeline is proven. It has no coordination/routing/memory/
lifecycle behavior and no natural-language capability. The "mini-amygdala coordinator" is a
future milestone; do not represent this artifact as such.
