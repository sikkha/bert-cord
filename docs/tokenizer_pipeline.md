# Tokenizer pipeline (Tokenizer Milestone)

A reproducible, config-driven pipeline to prepare a corpus, train a subword tokenizer, and
evaluate it. The selected tokenizer will later be **frozen** and reused for all BERT-Cord MLM
pretraining. The current goal is engineering robustness, **not** maximal tokenizer quality.

> Scope: tokenizer artifact only. No model / coordination / language-understanding claim, and
> not a Transformers `AutoModel`. Tokenizers load via Hugging Face `tokenizers` /
> `PreTrainedTokenizerFast`.

## Overview

```
raw text ──▶ prepare_tokenizer_corpus.py ──▶ data/tokenizer_corpus/  (shards + manifest + report)
                                                    │
                                                    ▼
                      train_tokenizer.py --config configs/tokenizer/<algo>.yaml
                                                    │
                                                    ▼
                       artifacts/tokenizers/<name>/  (tokenizer.json + configs + manifest)
                                                    │
                                                    ▼
                            evaluate_tokenizer.py ──▶ evaluation.json + evaluation.md
```

## Install

```bash
python -m pip install -e ".[dev,train]"   # provides tokenizers, datasets, pyyaml
```

## 1. Prepare the corpus

Reads `.txt` / `.md` / `.jsonl` (dirs recursed) and/or a streamed Hugging Face dataset;
normalizes Unicode, drops empties, exact-deduplicates, deterministically shuffles, computes
per-script language stats, and writes shards + manifest + report.

```bash
python scripts/prepare_tokenizer_corpus.py \
  --input data/raw \
  --output-dir data/tokenizer_corpus \
  --normalization NFC --val-fraction 0.02 --shard-size 100000
```

Key options: `--normalization {NFC,NFKC,NFD,NFKD,none}`, `--no-dedup`, `--min-chars`,
`--shuffle-seed`, `--val-fraction`, `--shard-size`, `--text-field`, and the HF source flags
`--hf-dataset/--hf-config/--hf-split/--hf-max-docs`.

Outputs under `data/tokenizer_corpus/`:

```
train-00000.txt ...      one document per line (sharded)
validation.txt           held-out split
corpus_manifest.json     sources, counts, dedup stats, per-file SHA-256, language stats
corpus_report.md         human-readable summary
```

## 2. Train the tokenizer (config-driven)

Configs live in `configs/tokenizer/` (32k vocab; special tokens pinned to ids 0–4):

- `bert_cord_wordpiece_32k.yaml` — WordPiece (BERT-original)
- `bert_cord_byte_bpe_32k.yaml` — byte-level BPE (no true UNK)
- `bert_cord_unigram_32k.yaml` — Unigram + byte fallback (Metaspace; good for Thai)

```bash
python scripts/train_tokenizer.py \
  --config configs/tokenizer/bert_cord_unigram_32k.yaml \
  --input data/tokenizer_corpus/train-00000.txt \
  --output-dir artifacts/tokenizers \
  --corpus-manifest data/tokenizer_corpus/corpus_manifest.json
```

Config fields: `name`, `algorithm`, `vocab_size`, `min_frequency`, `normalization`,
`lowercase`, `byte_fallback` (unigram/bpe), `add_bert_postprocessor`, `model_max_length`,
`special_tokens` (must begin with `[PAD] [CLS] [SEP] [MASK] [UNK]` → ids 0–4).

A legacy single-file mode is preserved:
`python scripts/train_tokenizer.py --input corpus.txt --vocab-size 32000 --output artifacts/tokenizer.json`.

Outputs under `artifacts/tokenizers/<name>/`:

```
tokenizer.json            the tokenizer (HF tokenizers format)
tokenizer_config.json     PreTrainedTokenizerFast config (special tokens, max length)
special_tokens_map.json   special-token mapping
tokenizer_manifest.json   algorithm, vocab, special ids, corpus ref, git commit, SHA-256
README.md                 per-tokenizer model card
```

The trainer **verifies reserved-token integrity** (special tokens at ids 0–4) and fails if
violated.

## 3. Evaluate

```bash
python scripts/evaluate_tokenizer.py \
  --tokenizer artifacts/tokenizers/bert-cord-unigram-32k \
  --input data/tokenizer_corpus/validation.txt \
  --output-dir artifacts/tokenizers/bert-cord-unigram-32k
```

Metrics (written to `evaluation.json` + `evaluation.md`): unknown-token rate, avg tokens per
sentence, avg tokens per word, round-trip decode (exact + whitespace-normalized), vocabulary
utilization, and reserved-token integrity. Round-trip *exact* is naturally low for subword
tokenizers; the whitespace-normalized rate is the meaningful fidelity signal.

## Directory layout

```
configs/tokenizer/*.yaml            (tracked) tokenizer configs
src/coordinator_bert/corpus.py      (tracked) corpus preparation
src/coordinator_bert/tokenizer_train.py  (tracked) config-driven trainer
src/coordinator_bert/tokenizer_eval.py   (tracked) evaluation metrics
scripts/prepare_tokenizer_corpus.py (tracked)
scripts/train_tokenizer.py          (tracked, extended)
scripts/evaluate_tokenizer.py       (tracked)
data/raw/                           (git-ignored) raw inputs / sample corpus
data/tokenizer_corpus/              (git-ignored) prepared corpus
artifacts/tokenizers/<name>/        (git-ignored) trained tokenizer artifacts
```

Corpus and tokenizer **outputs are git-ignored** (`data/`, `artifacts/`); the frozen tokenizer
will be distributed as a release artifact, not committed to Git history.

## Corpus download

Real multilingual corpora (English/Thai Wikipedia, OSCAR, mC4, FineWeb, CC100) exceed 1 GB and
are **not** downloaded automatically — see [`recommended_corpus.md`](recommended_corpus.md) for
dataset ids, sizes, subsets, and exact DGX download commands.

## Expected outputs (sample corpus, tiny)

On the bundled ~2.6 KB sample corpus the vocab stays small (hundreds), all three algorithms
train with special tokens at ids 0–4, reserved-token integrity is OK, and byte-level BPE reports
0% unknown-token rate with ~100% whitespace-normalized round-trip. Real 32k vocabularies require
the large corpus above.
