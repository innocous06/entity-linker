# Limitations and Known Constraints

## 1. Candidate Generation Recall Ceiling

The inverted index blocking strategy relies on at least one shared informative token between primary and target entities. In empirical evaluation on the full 2.5M entity benchmark:
- **Captured by index**: 97.8% of ground truth pairs contain at least one shared token.
- **Ranked within top-50/source**: 84.5% - 88.0% of ground truth pairs are ranked in the top-50 window.
- **Ranked within top-100/source**: 91.1% of ground truth pairs are ranked in the top-100 window.

Because downstream classification operates strictly on the emitted candidate set, ground-truth matches outside the top-k window are permanently lost. Under an 88% blocking recall, the mathematical upper bound for Macro $F_{0.5}$ is 0.973 even with 100% classifier precision.

## 2. Rule-Based Transliteration Scope

The inverse synthetic normalizer incorporates specific heuristics for Latin-transliterated business records (consonant doubling, phonetic spelling of suffixes like "elelpii" and "praaivett limittedd"). 

**Unresolved scenarios**:
- Non-Latin character scripts (e.g. Devanagari, Bengali, Tamil) passed without Latin transliteration.
- Extreme phonetic distortion where every token undergoes multiple phoneme shifts simultaneously (e.g. "Orchid Renewable" $\to$ "Smt BRIXUMBRABELO").
- Pure acronym-to-word conversions without lexical overlap.

## 3. Scale and Hardware Constraints

- **Execution Time**: On a single CPU core, querying 1M primary entities against 1M target entities requires ~30 minutes. Parallel execution across 4 physical cores reduces this to ~8 minutes.
- **Disk Space**: Streaming 100,000-pair chunks to disk requires approximately 2 GB - 4 GB of temporary scratch storage during the blocking phase. Scratch files are deleted immediately after DuckDB merging.
- **No Vector / Semantic Embeddings**: The current prototype does not utilize deep semantic sentence transformers (e.g. BGE-M3 or MiniLM) due to inference latency constraints on 5M+ pairs in resource-constrained environments.

## 4. Single-Country Partitioning Assumption

The pipeline partitions candidate generation strictly by country (`India`, `US`, `France`). While this matches the competition protocol where entities do not cross national borders, real-world multinational entity resolution requires cross-border linking (e.g. matching a US parent corporation to an Indian subsidiary).
