# Recommended tokenizer-training corpus (download on the DGX)

The appropriate multilingual corpora for a BERT-Cord tokenizer (English + Thai + technical /
markdown / code) are **all larger than 1 GB**, so per project policy they were **not downloaded
here**. Build the real corpus directly on the DGX server using the commands below, then run the
pipeline in `docs/tokenizer_pipeline.md`.

> A tiny hand-made sample corpus exists locally at `data/raw/samples/` (git-ignored) purely so
> the pipeline is runnable/testable offline. It is **not** a training corpus.

Priority order: **1) English · 2) Thai · 3) technical documentation · 4) markdown / JSON / code.**

## Candidate datasets

| # | dataset (HF id) | languages | est. full size | suggested subset |
|---|---|---|---|---|
| 1 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.en` | English | ~20 GB | stream 1–2M docs |
| 2 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.th` | Thai | ~1–2 GB | stream 0.3–0.5M docs |
| 3 | [`wikitext`](https://huggingface.co/datasets/wikitext) `wikitext-103-raw-v1` | English | ~180 MB (≤1 GB) | full |
| 4 | [`cc100`](https://huggingface.co/datasets/cc100) `th` | Thai | ~8 GB | stream subset |
| 5 | [`oscar-corpus/OSCAR-2301`](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) `th` (gated) | Thai | large | stream subset |
| 6 | [`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) `sample-10BT` | English | very large | stream subset |
| 7 | [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol) | code/markdown | ~1–2 GB | markdown/docs subset |
| 8 | [`codeparrot/github-code`](https://huggingface.co/datasets/codeparrot/github-code) | code/markdown | very large | stream subset |

Recommended starting mix (bounded, reproducible): **Thai Wikipedia (#2) + English Wikipedia
subset (#1) + wikitext-103 (#3) + a markdown/code slice (#7)**. Cap each source with
`--hf-max-docs` so total stays manageable.

## Build a bounded corpus with the project pipeline (streams; caps document counts)

Each command streams from HF and writes a prepared corpus; run them into the **same** output
dir to accumulate, or into separate dirs and pass all shards to the trainer.

```bash
# English Wikipedia (capped)
python scripts/prepare_tokenizer_corpus.py \
  --hf-dataset wikimedia/wikipedia --hf-config 20231101.en --hf-split train \
  --hf-max-docs 1000000 --text-field text \
  --output-dir data/tokenizer_corpus_en

# Thai Wikipedia (capped)
python scripts/prepare_tokenizer_corpus.py \
  --hf-dataset wikimedia/wikipedia --hf-config 20231101.th --hf-split train \
  --hf-max-docs 400000 --text-field text \
  --output-dir data/tokenizer_corpus_th

# English wikitext-103 (small; full)
python scripts/prepare_tokenizer_corpus.py \
  --hf-dataset wikitext --hf-config wikitext-103-raw-v1 --hf-split train \
  --text-field text --output-dir data/tokenizer_corpus_wikitext

# Markdown / code slice
python scripts/prepare_tokenizer_corpus.py \
  --hf-dataset bigcode/the-stack-smol --hf-split train \
  --hf-max-docs 200000 --text-field content \
  --output-dir data/tokenizer_corpus_code
```

Then train on all shards at once:

```bash
python scripts/train_tokenizer.py \
  --config configs/tokenizer/bert_cord_unigram_32k.yaml \
  --input data/tokenizer_corpus_en data/tokenizer_corpus_th \
          data/tokenizer_corpus_wikitext data/tokenizer_corpus_code \
  --output-dir artifacts/tokenizers
```

## Full raw downloads (alternative; large — DGX only)

```bash
pip install -U "huggingface_hub[cli]" datasets
# Full Thai Wikipedia (~1–2 GB):
huggingface-cli download wikimedia/wikipedia --repo-type dataset \
  --include "20231101.th/*" --local-dir data/raw/wikipedia_th
# Full wikitext-103 (~180 MB):
huggingface-cli download wikitext --repo-type dataset \
  --include "wikitext-103-raw-v1/*" --local-dir data/raw/wikitext103
```

> Notes: some datasets (e.g. OSCAR) are **gated** and require `huggingface-cli login`. The
> installed `datasets 5.0.0` streaming path had a URI quirk for the bare `wikitext` id in this
> sandbox; on the DGX, pin a working `datasets` version or use the `huggingface-cli download`
> path above. Estimated sizes are approximate — always confirm on the DGX before a full pull.

## Size policy applied here

All candidate real corpora exceed the 1 GB auto-download threshold, so **nothing large was
downloaded**. Only the ~2.6 KB local sample corpus exists. Perform the real download on the DGX
with the commands above.
