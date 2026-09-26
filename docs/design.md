# System Design & Architecture

## 1. Architecture Overview

The system is a two-stage entity resolution pipeline designed for large-scale multilingual business registries:
1. **Stage 1: Candidate Generation (Blocking)**: Out-of-core, streaming inverted index using Inverse Document Frequency (IDF) scoring to reduce an $N \times M$ search space (up to $2.2\text{M} \times 2.5\text{M} = 5.5 \times 10^{12}$ comparisons) down to $\le 50$ high-probability candidate pairs per entity.
2. **Stage 2: Pairwise Classification & Ranking**: A 26-feature pairwise gradient boosted decision tree (LightGBM) trained with GroupShuffleSplit validation and threshold calibration to optimize the Macro $F_{0.5}$ metric.

```mermaid
flowchart TD
    S1[Source 1: Primary Entities] --> NORM[Normalization & Inverse Transliteration]
    S2[Target Sources: S2 & S3] --> NORM

    NORM --> IDF[IDF Inverted Index Construction]
    IDF --> BLOCK[Streaming Out-of-Core Blocker]
    BLOCK -->|Chunks <= 100k| DISK[Temporary Parquet Chunks]
    DISK --> DUCK[DuckDB Merge & Streaming Top-k]
    DUCK --> CANDS[Candidate Pairs]

    CANDS --> FEAT[26-Feature Pairwise Extractor]
    FEAT --> MODEL[LightGBM Binary Classifier]
    MODEL --> CALIB[Threshold Calibration: Maximize Macro F0.5]
    CALIB --> PRED[Final Predictions: matching_results.tsv]
```

## 2. Invariants

The pipeline enforces the following system invariants:
1. **Memory Bound**: Peak heap RAM consumption must remain $\le 6\text{ GB}$ regardless of total input rows. No in-memory list or dictionary may hold $>250,000$ records concurrently; large sets are flushed to disk.
2. **Zero Leakage**: Validation splits must use `GroupShuffleSplit` on `s1_id`. No entity present in the training set may appear in the validation or test evaluation partitions.
3. **Deterministic Output**: Execution with fixed seeds produces byte-identical Parquet files and TSV outputs.
4. **Valid Formats**: Output TSV files use `quote_style='never'` with tab delimiters, explicit headers, and no trailing empty quoted strings (`""`).

## 3. Data Flow

1. **Ingestion**: Polars reads raw TSV tables with `truncate_ragged_lines=True` to prevent parsing crashes on malformed external text.
2. **Pre-processing**:
   - `clean_name`: Strips legal suffixes, web domains, and honorifics; collapses duplicated letters (`(.)\1+` $\to$ `\1`).
   - `get_address_tokens`: Extracts alphanumeric unit identifiers and house numbers with length $\ge 2$.
3. **Partitioned Indexing**: Tables are partitioned by country (`India`, `US`, `France`) to eliminate cross-country candidate generation.
4. **Candidate Emission**: Top 50 candidates are selected per source partition (`source2`, `source3`) independently to prevent inter-source crowding.
5. **Feature Matrix**: Pairwise feature computation is vectorized over chunks, yielding $N \times 26$ float32 matrices.
6. **Inference**: Probability scoring followed by singleton fallback (if $\max(\text{prob}) < \text{threshold}$, output is empty string).

## 4. Failure Modes and Mitigations

| Failure Mode | Root Cause | System Mitigation |
| :--- | :--- | :--- |
| **Out of Memory (OOM)** | Accumulating 70M+ candidate tuples in Python heap | Disk-backed streaming chunks (`CHUNK_FLUSH_SIZE = 100_000`); out-of-core DuckDB merge. |
| **Positional Truncation** | Capping inverted index posting lists at arbitrary sizes | Uninformative tokens exceeding document frequency thresholds (`MAX_NAME_DOCS = 12000`) are deleted from the index; informative tokens retain 100% of occurrences. |
| **Ragged / Dirty TSV Rows** | Unescaped tabs or line breaks in raw company addresses | Polars `truncate_ragged_lines=True` prevents parser aborts. |
| **Cross-Entity Validation Leak** | Random split placing matches of the same S1 in train and val | `GroupShuffleSplit` strictly partitions by primary entity ID (`s1_id`). |
| **Metric Misalignment** | Optimizing default logloss (0.50 cutoff) under class imbalance | Calibrated grid search on validation entities strictly maximizing Macro $F_{0.5}$. |
| **Single-Source Candidate Crowding** | Merging S2 and S3 before top-k ranking | Candidate pools are formed per target source independently before merging. |
