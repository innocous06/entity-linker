# entity-linker

[![Status: Active Prototype](https://img.shields.io/badge/STATUS-BETA_PROTOTYPE-c9654a?style=for-the-badge)](https://github.com/innocous06/entity-linker)
[![Language: Python](https://img.shields.io/badge/LANGUAGE-PYTHON_3.10+-18181f?style=for-the-badge)](https://www.python.org/)
[![Engine: DuckDB Out-of-Core](https://img.shields.io/badge/ENGINE-DUCKDB_OUT--OF--CORE-2e4c23?style=for-the-badge)](https://duckdb.org/)
[![License: MIT](https://img.shields.io/badge/LICENSE-MIT-18181f?style=for-the-badge)](LICENSE)

> Out-of-core multilingual entity resolution engine in Python. Engineered to resolve multi-source enterprise business registries under transliteration noise, letter doubling, and strict memory limits (<4 GB RAM) using streaming IDF-weighted inverted index blocking, pairwise feature extraction, and calibrated LightGBM classification.

---

## 1. Real Pipeline Execution Output

```text
$ python run_demo.py
=================================================================
  Multilingual Entity Resolution - End-to-End Demonstration
=================================================================

[STEP 1/2] Training & Validation Evaluation...

--- Running Blocking for TRAIN ---
  [France/source2] Emitted 86 candidate pairs -> cand_train_France_source2.parquet
  [France/source3] Emitted 85 candidate pairs -> cand_train_France_source3.parquet
  [India/source2] Emitted 707 candidate pairs -> cand_train_India_source2.parquet
  [India/source3] Emitted 635 candidate pairs -> cand_train_India_source3.parquet
  [US/source2] Emitted 315 candidate pairs -> cand_train_US_source2.parquet
  [US/source3] Emitted 231 candidate pairs -> cand_train_US_source3.parquet
[OK] Merged 2,059 candidate pairs into train_candidates.parquet
Extracting features for 2,059 candidate pairs...
Dataset summary: 2,059 pairs, 156 positive matches (7.58%)

--- Validation Results ---
Optimal Threshold: 0.25
Validation Precision: 0.6991
Validation Recall:    0.7500
Validation Macro F0.5: 0.6997

[STEP 2/2] Generating Test Predictions...

--- Running Blocking for TEST ---
  [India/source2] Emitted 14 candidate pairs -> cand_test_India_source2.parquet
  [India/source3] Emitted 14 candidate pairs -> cand_test_India_source3.parquet
  [US/source2] Emitted 7 candidate pairs -> cand_test_US_source2.parquet
  [US/source3] Emitted 5 candidate pairs -> cand_test_US_source3.parquet
  [France/source2] Emitted 2 candidate pairs -> cand_test_France_source2.parquet
  [France/source3] Emitted 2 candidate pairs -> cand_test_France_source3.parquet
[OK] Merged 44 candidate pairs into test_candidates.parquet
Extracting features for 44 test candidate pairs...
[OK] Generated matching_results.tsv (15 rows)
[OK] Generated candidate_pairs.tsv (44 rows)

=================================================================
  Demonstration Complete.
  Validation F0.5: 0.6997
  Outputs saved in: D:\code\entity-linker\workspace\output
=================================================================
```

---

## 2. Overview & Problem Statement

Entity resolution matches records across disparate databases that represent the same real-world entity without common unique keys. In large enterprise datasets (2.2M primary records against two 2.5M target tables), naive pairwise comparison requires over 5.5 trillion comparisons.

The problem is complicated by:
1. **Multilingual Transliteration Noise**: Non-English company names undergoing synthetic Latin transliteration, resulting in letter-doubling ("Dream Construction" -> "ddriim knsttrkssn"), phonetic substitution ("Silver" -> "silvr"), and legal suffix expansions ("LLP" -> "elelpii", "Private Limited" -> "praaivett limittedd").
2. **Memory Constraints**: Generating candidates across millions of entities can easily exceed 40 GB of RAM if kept in memory or transferred via inter-process communication (IPC).
3. **Severe Class Imbalance**: With ~3.5 true matches per entity against millions of candidates, false positives heavily penalize the competition target metric, Macro F0.5.

`entity-linker` solves this with an out-of-core streaming inverted index blocker weighted by Inverse Document Frequency (IDF) and a 26-feature pairwise gradient boosted classifier tuned for precision-biased F0.5 optimization.

---

## 3. Engineering Trade-Off: Unconstrained SOTA vs. Memory-Optimized Architecture

### What Could Be Done in an Unconstrained Setting (Local GPU Rig)

In a hardware environment with unconstrained RAM (64 GB - 128 GB) and dedicated NVIDIA GPUs:
1. **Dense Bi-Encoder Retrieval**: Fine-tuning a multilingual Sentence Transformer (e.g. `BGE-M3` or `multilingual-e5`) to encode business names and addresses into 1024-dimensional vectors, indexing 5M records in a GPU-accelerated FAISS/HNSW index. This provides semantic matching even when lexical overlap is zero (e.g. `Orchid Renewable` -> `Smt BRIXUMBRABELO`), pushing candidate recall past 98%.
2. **Cross-Encoder Transformer Reranker**: Passing top 20 candidates per entity to a Cross-Encoder (e.g. `DeBERTa-v3` or `BGE-Reranker-Large`) computing full all-to-all attention across primary and candidate tokens, pushing precision past 99%.
3. **Global Bipartite Graph Matching**: Solving maximum-weight bipartite matching across candidate probability graphs to eliminate multi-target duplicate conflicts.

### The Memory Problem & Real-World Bottlenecks

In cloud developer environments or memory-bounded servers (<40 GB RAM, CPU-only):
- Storing 5M dense 1024-dimensional float32 vectors requires over 20 GB of RAM/VRAM for embeddings alone.
- Generating 70M+ candidate pairs in Python heap and serializing them across multiprocessing workers via IPC caused immediate Out-Of-Memory (OOM) crashes (>45 GB RAM consumption).
- Ephemeral cloud instances without 24 GB VRAM GPUs make dense neural reranking across millions of candidate pairs computationally prohibitive.

### How I Optimized to Counter the Memory Problem

To deliver a functional, high-throughput solution under a strict memory envelope (<4 GB RAM) without specialized hardware:
1. **Out-of-Core Chunk Streaming**: Replaced in-memory candidate lists with streaming batches that flush 100,000 candidate pairs directly to Snappy-compressed Parquet chunks on disk. Chunks are merged via DuckDB out-of-core columnar execution, bounding peak RAM to **< 3.2 GB**.
2. **IDF-Weighted Token Indexing**: Rather than flat count matching, precomputed token Inverse Document Frequency (`IDF = ln((N + 1) / (df + 1))`). High-frequency uninformative words are deleted from the index, eliminating positional bias and breaking score ties to reach 91.1% recall.
3. **Inverse Transliteration De-Noising**: Replaced heavy neural models with deterministic regex transforms (vowel/consonant deduplication `(.)\1+` -> `\1`, phonetic legal suffix mapping `elelpii` -> `LLP`, building unit identifier extraction) that collapse corrupted strings to canonical forms upfront with zero GPU compute.
4. **Group-Aware Tabular GBDT**: Extracted 26 lightweight pairwise lexical, phonetic, and geographical features scored with LightGBM and calibrated decision thresholds.

---

## 4. Quick Start

### Installation

```bash
git clone https://github.com/innocous06/entity-linker.git
cd entity-linker
pip install -r requirements.txt
```

### Run Demonstration

Run the complete pipeline on the included synthetic benchmark dataset (takes ~5 seconds):

```bash
python run_demo.py
```

### Run on Custom Datasets

```bash
# Run training and evaluation
python -m src.pipeline --data-dir path/to/tsv_folder --work-dir workspace --train

# Run inference and generate submission TSV files
python -m src.pipeline --data-dir path/to/tsv_folder --work-dir workspace --predict
```

---

## 5. Architecture & Data Flow

For detailed specifications, invariants, and failure matrices, see [`docs/design.md`](docs/design.md).

```text
Source 1 (Primary)            Target Sources (S2 & S3)
       │                                  │
       ▼                                  ▼
┌───────────────────────────────────────────────┐
│   Inverse Transliteration & Normalization     │  <-- Letter deduplication, synthetic suffix mapping
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│    IDF Inverted Index Construction (Target)   │  <-- Precomputed log((N+1)/(df+1)) token weights
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│    Streaming Out-of-Core Blocker (<4 GB RAM)  │  <-- Prunes tokens > 12k; disk-flushed chunks
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│     DuckDB Streaming Merge & Top-50 Ranking   │  <-- Separate per-source candidate emission
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│     26-Feature Pairwise Vector Extraction     │  <-- Lexical, phonetic, address unit, pincode
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│    LightGBM GBDT + F0.5 Calibrated Cutoff     │  <-- GroupShuffleSplit validation; optimal threshold
└──────────────────────┬────────────────────────┘
                       │
                       ▼
  [ matching_results.tsv ]   [ candidate_pairs.tsv ]
```

---

## 6. Key Engineering Decisions

Architectural alternatives and trade-offs are logged in [`docs/decisions.md`](docs/decisions.md).

1. **IDF Weighting over Flat Token Counts ([D-002](docs/decisions.md#d-002-idf-weighting-for-candidate-ranking-vs-flat-token-frequency-match))**: Common words ("traders", "solutions", "plaza") produce identical match scores for thousands of candidates. Precomputing Inverse Document Frequency (`IDF = ln((N + 1) / (df + 1))`) assigns rare tokens (e.g. "osarambagh") high weights (~12.0) and common tokens low weights (~5.0), eliminating arbitrary hash-order tie-breaking.
2. **Streaming Chunk Flushing over In-Memory IPC ([D-001](docs/decisions.md#d-001-streaming-inverted-index-with-disk-chunking-vs-in-memory-ipc-list-return))**: Returning 70M+ candidate tuples across multiprocessing workers via IPC caused out-of-memory crashes on 48 GB systems. Flushing 100,000-pair batches directly to Snappy-compressed Parquet chunks merged via DuckDB bounds peak RAM to **< 3.2 GB**.
3. **Independent Per-Source Blocking ([Bug 2](docs/bug-log.md#bug-2-global-top-k-squeeze-across-multiple-target-sources))**: Primary records average ~3.5 true matches distributed across both Source 2 and Source 3. Merging before candidate pruning caused noisy matches from one source to displace valid matches from the second. Blocking S1 x S2 and S1 x S3 independently preserves candidate diversity.
4. **GroupShuffleSplit Validation ([D-004](docs/decisions.md#d-004-groupshufflesplit-by-s1-entity-id-for-model-validation))**: Grouping validation splits strictly by primary entity ID (`s1_id`) prevents optimistic validation leakage and mirrors unseen test distribution.

---

## 7. Empirical Benchmark Results

Evaluated on 5,000 real primary records against 1,000,000 India and 1,535,000 US candidate records on an AMD Ryzen 7 7735HS Windows 11 system:

| Blocking Strategy | India S2 Recall | US S2 Recall | Weighted Overall | Throughput (1 CPU core) | Peak RAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline Name-Only Gating | 36.03% | ~60.0% | ~50.4% | ~350 qps | 2.8 GB |
| Capped Index (No Pruning) | 37.30% | 46.45% | 42.93% | ~180 qps | 4.1 GB |
| Fixed Index + Smart Addr (Top 50/src) | 84.47% | 87.16% | 86.08% | ~500 qps | 3.1 GB |
| Fixed Index + Smart Addr (Top 60/src) | 85.63% | 87.43% | 86.71% | ~500 qps | 3.2 GB |
| Fixed Index + Smart Addr (Top 100/src) | 87.54% | 88.41% | 88.06% | ~490 qps | 3.2 GB |
| **IDF Inverted Index (Top 60/src)** | **89.61%** | **~92.0%** | **91.04%** | **~480 qps** | **3.2 GB** |
| **IDF Inverted Index (Top 100/src)** | **91.10%** | **~93.0%** | **92.24%** | **~470 qps** | **3.2 GB** |

---

## 8. Documentation

- **System Design & Invariants**: [`docs/design.md`](docs/design.md)
- **Architecture Decisions**: [`docs/decisions.md`](docs/decisions.md)
- **Bug Post-Mortem Log**: [`docs/bug-log.md`](docs/bug-log.md)
- **Limitations & Mathematical Bounds**: [`docs/limitations.md`](docs/limitations.md)
- **Literature References**: [`docs/references.md`](docs/references.md)

---

## 9. Limitations

- **Candidate Recall Ceiling**: The inverted index achieves 88% - 91% recall at top-100 candidates. Matches omitted from the candidate set cannot be recovered by downstream classifiers.
- **Rule-Based Transliteration**: Normalization targets Latin-transliterated Indian, US, and French business registries. Un-transliterated scripts (e.g. Devanagari) require grapheme-to-phoneme models.
- **Single-Country Partitioning**: Candidate generation assumes entities do not match across country boundaries.

See [`docs/limitations.md`](docs/limitations.md) for complete analysis.

---

## 10. License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.
