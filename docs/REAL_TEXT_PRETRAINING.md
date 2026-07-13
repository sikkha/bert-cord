# Real-text MLM pretraining (packed corpus)

The real-language stage trains the 27.01M BERT-Cord encoder on a **pre-tokenized, packed**
token corpus instead of tokenizing text rows on the fly. This document explains why, the packed
format, and the DGX run sequence.

## Why article-level truncation was rejected

The previous HF-text path did:

```python
ds = load_dataset(name, config)
enc = tok.encode(row[text_column]); ids = enc.ids[:max_seq_length]   # keep only the first N
```

For long Wikipedia articles this is unacceptable:

- it **discards most tokens of every article** (only the first `max_seq_length` survive), so the
  model never sees the bulk of the corpus;
- it **materializes the whole encoded corpus as Python lists** in memory.

The packed pipeline instead keeps *all* useful content by splitting each document across as many
fixed-length sequences as needed, and stores token IDs as memory-mapped numpy arrays.

## Offline packing design

`scripts/tokenize_and_pack_corpus.py` (→ `src/coordinator_bert/packed_corpus.py`) writes:

```
data/tokenized/<run>/
  manifest.json
  train/shard-00000.npy ...        int array [num_sequences, sequence_length]
  validation/shard-00000.npy ...
```

- dtype **uint16** when `vocab_size <= 65535`, else **uint32** (32k vocab → uint16).
- Each row is `[CLS] content... [SEP] PAD...` — **token IDs only**.
- Content is encoded **without** automatic special tokens, then framed explicitly, so `[CLS]` /
  `[SEP]` are never double-inserted.
- One input **line = one document**; content is chunked to `sequence_length - 2` tokens; a
  sequence **never crosses a document boundary** (v1); only the **final chunk** of a document is
  padded; empty documents are skipped.
- Deterministic (sorted-file, in-file order), bounded memory (shards flushed at
  `--sequences-per-shard`), **atomic** shard writes (temp file + rename).
- Fails loudly if the tokenizer vocab size or special-token IDs
  (`[PAD]=0,[CLS]=1,[SEP]=2,[MASK]=3,[UNK]=4`) do not match expectations.
- The **manifest** records: format version, tokenizer path + SHA-256 + vocab size, source files +
  SHA-256 + detected language, git commit, sequence length, dtype, packing policy, train/val
  sequence counts, per-shard names/shapes/bytes/SHA-256, special-token IDs, timestamp, seed, and
  counters (documents read/skipped, source tokens, packed sequences, padding tokens, packing
  efficiency, unknown-token count/rate).

**No MLM masks or labels are precomputed or stored.** Masking remains **dynamic** in the
collator.

## Dynamic masking (unchanged objective)

`PackedTokenDataset` memory-maps the shards and returns one token-ID row per index (no
whole-corpus Python lists). The packed DataLoader uses the existing `MLMasker` via
`PackedMLMCollator`, which:

- derives `attention_mask` from the pad id (rows are pre-padded),
- never selects special tokens (incl. PAD) for masking,
- applies the standard 15% / 80-10-10 corruption with `-100` labels for unmasked positions.

The MLM mathematical objective is **identical** to the synthetic/HF paths. Validation masking is
reproducible across runs (fixed generator seed), matching the existing paths.

## Config dispatch

`DataConfig.packed_dataset_dir` selects the packed loader. Dispatch priority in
`build_dataloaders`: **`packed_dataset_dir` → `dataset_name` (HF text) → synthetic**. Existing
synthetic and HF paths are unchanged.

## Build the packed corpus (on the DGX)

First train/select a **32k byte-BPE tokenizer** (see `docs/tokenizer_pipeline.md`) and prepare a
bounded EN+TH corpus (see `docs/recommended_corpus.md`). Then pack:

### 128-token pilot corpus

```bash
python scripts/tokenize_and_pack_corpus.py \
  --tokenizer artifacts/tokenizers/bert-cord-byte-bpe-32k \
  --train-input data/tokenizer_corpus_en data/tokenizer_corpus_th \
  --validation-input data/tokenizer_corpus_en/validation.txt \
  --output-dir data/tokenized/bert_cord_en_th_128_v1 \
  --sequence-length 128 --sequences-per-shard 100000 --seed 42

python scripts/validate_packed_corpus.py data/tokenized/bert_cord_en_th_128_v1 \
  --tokenizer artifacts/tokenizers/bert-cord-byte-bpe-32k --require-validation
```

### 512-token full corpus

```bash
python scripts/tokenize_and_pack_corpus.py \
  --tokenizer artifacts/tokenizers/bert-cord-byte-bpe-32k \
  --train-input data/tokenizer_corpus_en data/tokenizer_corpus_th data/tokenizer_corpus_code \
  --validation-input data/tokenizer_corpus_en/validation.txt \
  --output-dir data/tokenized/bert_cord_en_th_512_v1 \
  --sequence-length 512 --sequences-per-shard 100000 --seed 42

python scripts/validate_packed_corpus.py data/tokenized/bert_cord_en_th_512_v1 \
  --tokenizer artifacts/tokenizers/bert-cord-byte-bpe-32k --require-validation
```

## Run sequence & acceptance gates

Run in order; each gate must pass before the next.

1. **Smoke** (100 steps, 128-token corpus) — proves the packed path end-to-end:
   ```bash
   python scripts/pretrain_mlm.py --config configs/experiments/dgx_real_text_smoke.yaml
   ```
   Gate: no runtime error; resolved runtime printed (device cuda, precision bf16); loss finite
   and generally decreasing; checkpoint written to `experiments/dgx_real_text_smoke/checkpoints`;
   W&B offline run dir + sync command printed.

2. **Checkpoint/resume test** — resume the smoke run from its checkpoint root:
   ```bash
   python scripts/pretrain_mlm.py --config configs/experiments/dgx_real_text_smoke.yaml \
     --max-steps 150 --resume experiments/dgx_real_text_smoke/checkpoints
   ```
   Gate: `latest.json` resolved, checksum verified, global step restored, no loss spike.

3. **Pilot** (1000 steps, 128-token corpus) — **fresh initialization** by default:
   ```bash
   python scripts/pretrain_mlm.py --config configs/experiments/dgx_real_text_pilot.yaml
   ```
   Gate: stable training, decreasing val loss/perplexity, sane masked accuracy trend; analyze
   with `scripts/analyze_training_curve.py` on the metrics.

4. **Full** (512-token corpus) — **only after** the above, with a deliberately chosen step count
   (the config's `max_steps` is a **placeholder**):
   ```bash
   python scripts/pretrain_mlm.py --config configs/experiments/dgx_real_text_full.yaml \
     --max-steps <chosen>
   ```
   Full training is **never launched by tests/CI**.

## Fresh initialization requirement

Pilot and full runs start from **fresh weights** unless you explicitly pass
`--resume <checkpoint-root>`. The pretrainer only resumes when `--resume` is given, so omitting it
guarantees fresh initialization. Do not resume a 512-token full run from a 128-token pilot
checkpoint expecting comparable behavior — the sequence length (positional usage) differs.

## Tokenizer checksum requirement

The packed manifest stores the tokenizer's SHA-256. **Always** validate with `--tokenizer` so the
corpus is provably packed with the intended, frozen tokenizer:

```bash
python scripts/validate_packed_corpus.py <packed-dir> --tokenizer <tokenizer-dir> --require-validation
```

Re-training the tokenizer changes the checksum; a mismatch means the packed corpus and the
tokenizer are out of sync — repack before pretraining.

## Artifact & disk-size expectations

- Packed size ≈ `num_sequences × sequence_length × bytes_per_id` (2 for uint16). Example: 10M
  sequences × 512 × 2 bytes ≈ **~10 GB**; the 128-token pilot is ~4× smaller per sequence.
- `data/tokenized/` and `experiments/` are **git-ignored** — packed corpora and checkpoints are
  never committed. Distribute large artifacts via release channels, not Git.
- Packing efficiency (content / total slots) is reported in the manifest; short final chunks and
  short documents lower it. Longer `sequence_length` typically improves efficiency on long
  articles.

> Scope: this is MLM pretraining on real multilingual text. It makes no coordination, routing,
> or language-understanding claim beyond masked-token modeling, and does not change the model
> architecture or the MLM objective.
