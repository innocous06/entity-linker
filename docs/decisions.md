# Architecture Decision Records

## D-001: Streaming Inverted Index with Disk Chunking vs In-Memory IPC List Return
Date: 2026-09-26
Context: Generating candidate pairs across 2.2M Source 1 entities and 2.5M Target entities (Source 2 and 3) produces 150M+ candidate pairs. Holding these in Python lists or passing them across multiprocessing workers via IPC caused out-of-memory crashes on systems with 48 GB RAM.
Options considered:
1. Multiprocessing Pool with in-memory list concatenation.
2. Ray/Dask distributed in-memory dataframes.
3. Streaming single-process or multi-worker disk chunking flushing 100,000 pairs to Snappy-compressed Parquet files, merged via DuckDB out-of-core engine.
Decision: Option 3.
Why: Guaranteed memory bound (<4 GB peak RAM) regardless of dataset scale. DuckDB streams Parquet chunks without memory pressure.
Trade-off: Minor disk I/O overhead compared to pure in-memory execution.
Revisit if: Dedicated bare-metal machine with 256 GB+ RAM is available.

## D-002: IDF Weighting for Candidate Ranking vs Flat Token Frequency Match
Date: 2026-09-26
Context: In 1M+ business entity tables, common words (e.g. "center", "traders", "solutions") produce identical match scores for hundreds of candidates, causing Python sort ties to arbitrarily displace true matches outside the top-k window.
Options considered:
1. Flat scoring (+3 for name, +2 for address) with arbitrary tie-breaking.
2. Precomputed Inverse Document Frequency (IDF = ln((N + 1) / (df + 1))) applied to each token match.
Decision: Option 2.
Why: Rare tokens (e.g. "kalyan", "osarambagh") receive high weight (~12.0) while common tokens receive low weight (~5.0). Empirical benchmark on 2,000 real entities showed recall jumped from 84.5% to 91.1% at top-100 purely from tie-breaking resolution.
Trade-off: Requires pre-computing document frequency across the target partition before querying.
Revisit if: Full dense embedding search (e.g. FAISS/ScaNN) is introduced.

## D-003: Inverse Synthetic De-Noising vs Raw Fuzzy String Distance
Date: 2026-09-26
Context: Target entities exhibit systematic synthetic corruption (vowel doubling "ddriim", consonant transliteration "knsttrkssn", legal suffix phonetic expansion "elelpii"). Standard string distances fail when multiple consecutive substitutions degrade similarity.
Options considered:
1. Raw Levenshtein/Jaro-Winkler on unaltered strings.
2. Inverted synthetic normalization: letter deduplication ((.)\1+ -> \1), regex canonicalization of synthetic legal suffixes, consonant skeleton matching, and house-number preserving address tokenization.
Decision: Option 2.
Why: Reduces distorted strings to common canonical forms upfront. Address token recall doubled from 36.0% to 75.2% on benchmark data.
Trade-off: Domain-specific rules tuned to Latin-transliterated Indian, US, and French naming conventions.
Revisit if: Processing un-transliterated non-Latin scripts (e.g. Devanagari, Cyrillic) requiring phonemic G2P models.

## D-004: GroupShuffleSplit by S1 Entity ID for Model Validation
Date: 2026-09-26
Context: Randomly splitting candidate pairs into train and validation sets leaks S1 entity patterns and causes optimistic validation scores that fail on unseen entities.
Options considered:
1. Random train_test_split on candidate pairs.
2. GroupShuffleSplit grouped by s1_id (80% train entities, 20% validation entities).
Decision: Option 2.
Why: Ensures the validation set contains strictly unseen businesses, mirroring the test evaluation protocol.
Trade-off: Slightly smaller effective training sample size for individual rare entity classes.
Revisit if: Stratified k-fold cross-validation is needed across countries.

## D-005: Threshold Calibration for Macro F0.5 Optimization
Date: 2026-09-26
Context: Default classifier decision boundary (0.50) is calibrated for accuracy or balanced F1. Macro F0.5 weights precision twice as heavily as recall (beta = 0.5), penalizing false positive entity links.
Options considered:
1. Default 0.50 probability cutoff.
2. Grid search over thresholds [0.20, 0.85] measuring exact Macro F0.5 per competition specifications.
Decision: Option 2.
Why: Optimal operating threshold shifts depending on class imbalance (~1:15 positive-to-negative ratio). Calibration yields 5-10% higher F0.5 on held-out entities.
Trade-off: Requires computing entity-level macro metrics during hyperparameter search.
Revisit if: Objective switches to F1 or F2.

## D-006: Out-of-Core Tabular Streaming vs. Dense Neural Embeddings under Memory Limits
Date: 2026-09-27
Context: Dense bi-encoder transformers (BGE-M3, multilingual-e5) paired with GPU-accelerated FAISS search deliver higher recall on extreme phonemic shifts. However, storing 5M 1024-dimensional float32 embeddings requires >20 GB RAM for vectors alone, and generating 70M+ candidate pairs across multiprocessing workers caused instant out-of-memory crashes on cloud runtimes with 48 GB limits.
Options considered:
1. Dense bi-encoder vector embeddings with FAISS/HNSW index.
2. In-memory multiprocessing inverted index.
3. Out-of-core streaming inverted index with IDF weighting, disk-backed chunk flushing, and DuckDB merging.
Decision: Option 3.
Why: Guaranteed memory bound (<3.2 GB peak RAM) allowing reliable execution on standard developer laptops and cloud instances. High throughput (~500 queries/sec per CPU core) without GPU hardware dependencies.
Trade-off: Misses extreme multi-token phonemic shifts where lexical overlap is 0%, capping candidate recall at ~91% vs. theoretical 98%+ with dense vectors.
Revisit if: Dedicated workstation with 64 GB+ RAM and 24 GB VRAM GPU is available.
