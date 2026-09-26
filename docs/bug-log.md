# Bug Log

### Bug 1: Positional Bias via Inverted Index Capping Without Deletion
- Found by: Empirical recall evaluation on full dataset.
- Trigger: Processing tables with >1M rows when using `if len(idx[t]) < MAX: idx[t].add(eid)`.
- Symptom: Candidate recall dropped to 42.93% on 2.5M rows despite testing 83% on small slices.
- Root cause: High-frequency tokens reached capacity (e.g. 3,000) within the first 30,000 rows. The index rejected all records after row 30,000, creating severe positional bias.
- Fix: When token frequency exceeds threshold, delete the token entirely (`del idx[t]`) and record it in a pruned set so no entity matches an uninformative token. Informative tokens retain all occurrences regardless of file position.
- Regression test: `scripts/test_index_positional_bias.py` verifying uniform hit rates across beginning, middle, and end chunks.
- Lesson: Any size capping on inverted index entries must discard the key, not freeze document addition.

### Bug 2: Global Top-k Squeeze Across Multiple Target Sources
- Found by: Database query breakdown of recalled candidate sources.
- Trigger: Applying `ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY score DESC) < 40` across candidates from Source 2 and Source 3 combined.
- Symptom: Source 2 matches displaced Source 3 matches, missing valid multi-source linkages.
- Root cause: S1 businesses frequently match distinct entities in both Source 2 and Source 3 (average ~3.5 true matches per S1). Merging before top-k selection caused noisy matches from one source to crowd out valid matches from the second.
- Fix: Block each target source independently (e.g. top 50 from Source 2, top 50 from Source 3) and merge candidates only after source-level ranking.
- Regression test: Evaluated in `run_blocking` with separate per-source output partitions.
- Lesson: Multi-source record linkage must preserve source diversity before final scoring.

### Bug 3: Name-Only Gating Suppressing Address Token Matching
- Found by: Ground truth failure analysis on Indian business records.
- Trigger: Running blocking with `if not cand_scores: continue` before evaluating address tokens.
- Symptom: India S2 recall was capped at 36.03%.
- Root cause: Synthetic transliteration distorted 64% of Indian business names (e.g. "Dream Construction" -> "ddriim knsttrkssn"), resulting in zero name token matches. Address tokens (which were largely intact) were never checked.
- Fix: Allow address tokens to generate candidate scores independently, weighted by their respective IDF values.
- Regression test: Verified in local benchmark on 5,000 India S1 entities, increasing recall from 36.03% to 84.72%.
- Lesson: In dirty or synthetically corrupted datasets, secondary fields (addresses, tax IDs) must anchor candidates independently.

### Bug 4: Inter-Process Communication Memory Explosion on Raw Candidate Lists
- Found by: Kernel OOM crash in high-RAM execution environment.
- Trigger: Returning 72M tuples across 4 Python multiprocessing workers via `pool.map()`.
- Symptom: Python process working set spiked past 45 GB and kernel was killed.
- Root cause: Pickling and deserializing 72M strings across process boundaries duplicates memory multiple times.
- Fix: Stream records and flush batches of 100,000 pairs directly to Snappy-compressed Parquet files on disk, then merge via DuckDB out-of-core engine. Peak RAM dropped to <4 GB.
- Regression test: Verified end-to-end in `src/blocking.py`.
- Lesson: Never transfer tens of millions of Python objects through IPC queues; stream to disk buffers.

### Bug 5: Tokenizer Regex Stripping Alphanumeric House and Unit Numbers
- Found by: Address matching diagnostic on missed building identifiers.
- Trigger: Tokenizing addresses with `re.sub(r'[^a-z0-9\s]', ' ', s)` followed by `len(t) >= 3`.
- Symptom: House numbers like "16-11-23", "E-7", "28-B", and "#20-6-3" were split into 1-2 character tokens and deleted.
- Root cause: Punctuation stripping without preserving hyphenated unit numbers destroyed the most discriminative address tokens.
- Fix: Clean punctuation while preserving hyphens and retaining tokens containing digits if length >= 2.
- Regression test: Verified in `src/normalizer.py:get_address_tokens`.
- Lesson: Address tokenization must treat numbers with different length thresholds than alphabetical prose.
