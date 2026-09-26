# Unconstrained Hardware Architecture & Implementation Blueprint

This document specifies the technical design, models, data structures, and execution workflow for solving enterprise-scale entity resolution on unconstrained local hardware (64 GB - 128 GB RAM, dedicated NVIDIA GPU, NVMe PCIe Gen4 SSD) to reach **0.98+ Macro F0.5**.

---

## 1. Target Hardware Specifications

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **System Memory** | 64 GB DDR5-5600 | 128 GB DDR5-6000 |
| **GPU / VRAM** | 1x NVIDIA RTX 4080 (16 GB VRAM) | 1x NVIDIA RTX 4090 (24 GB VRAM) or 2x RTX 3090 |
| **Local Storage** | 1 TB PCIe Gen4 M.2 NVMe (>= 5,000 MB/s) | 2 TB PCIe Gen5 M.2 NVMe (>= 10,000 MB/s) |
| **CPU** | 8 Cores / 16 Threads (e.g. Ryzen 7 7700X) | 16 Cores / 32 Threads (e.g. Ryzen 9 7950X / Core i9) |

---

## 2. End-to-End System Pipeline

```text
               Source 1 (2.2M Primary Entities)
                             │
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Stage 0: Learned Character-Level Transliteration Inverter│  <-- ByT5-small seq2seq normalizer
 └───────────────────────────┬──────────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Stage 1: Dual Dense-Sparse Hybrid Candidate Retrieval    │
 │  ├─► Dense Bi-Encoder: Fine-Tuned BGE-M3 (1024-dim)      │  <-- GPU FAISS IndexHNSWFlat (24 GB VRAM)
 │  └─► Sparse BM25: Tantivy / DuckDB FTS Engine (CPU)      │  <-- Exact BM25 length-normalized scoring
 │  └─► Fusion: Reciprocal Rank Fusion (RRF, Top 30/src)    │  <-- Target Recall: >= 98.5%
 └───────────────────────────┬──────────────────────────────┘
                             │ Top-30 Candidates / Source
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Stage 2: Cross-Encoder Transformer Reranker              │
 │  - Input: [CLS] S1 [SEP] Candidate [SEP]                 │
 │  - Architecture: mDeBERTa-v3-base / BGE-Reranker-Large   │  <-- All-to-all cross-attention
 │  - Optimization: TensorRT-LLM FP16 Batch Inference       │  <-- Target Precision: >= 99.0%
 └───────────────────────────┬──────────────────────────────┘
                             │ Candidate Probability Matrix
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Stage 3: Global Bipartite Graph Matching                 │
 │  - Maximum-Weight Bipartite Linkage Assignment           │  <-- Resolves conflicting multi-matches
 │  - Dynamic Singleton Cutoff Calibration                  │
 └───────────────────────────┬──────────────────────────────┘
                             │
                             ▼
              Final High-Precision Submissions
```

---

## 3. Component Deep Dive

### Phase 0: Learned Character-Level De-Noiser (ByT5-Small)
Synthetic transliteration introduces letter doubling (`ddriim` -> `dream`), phoneme substitution (`knsttrkssn` -> `construction`), and acronym phonetic spellings (`elelpii` -> `LLP`).
- **Model**: `google/byt5-small` (byte-level token-free sequence-to-sequence model).
- **Training Objective**: Trained on ground-truth paired text discrepancies from Source 1 and Source 2/3.
- **Inference**: High-throughput ONNX Runtime beam-1 search. Normalizes distorted strings into canonical forms prior to embedding.

### Phase 1: Hybrid Retrieval Engine (Dense + Sparse)
Dense embeddings capture phonemic/semantic relationships with 0% lexical overlap, while BM25 ensures rare numeric codes (pincodes, plot numbers) are preserved.

1. **Dense Bi-Encoder**:
   - Base Architecture: `BAAI/bge-m3` or `intfloat/multilingual-e5-base`.
   - Loss Function: `MultipleNegativesRankingLoss` (InfoNCE) with in-batch negatives plus hard negatives mined from top BM25 false positives.
   - Text Representation: `passage: {business_name} | {business_address} | {country}`.
   - Index: GPU FAISS `IndexHNSWFlat` with $M = 32, \text{efSearch} = 64$.
2. **Sparse Retrieval (BM25)**:
   - Built on Tantivy (Rust-based Lucene engine) or DuckDB FTS.
   - Exact BM25 parameters: $k_1 = 1.2, b = 0.75$.
3. **Reciprocal Rank Fusion (RRF)**:
   $$\text{RRF\_Score}(d) = \frac{1}{60 + r_{\text{dense}}(d)} + \frac{1}{60 + r_{\text{sparse}}(d)}$$
   Emits top-30 candidates per target source independently.
   **Expected Candidate Recall**: $\mathbf{\ge 98.5\%}$.

### Phase 2: Cross-Encoder Transformer Reranker
Pairwise GBDT on hand-crafted features misses complex cross-token interactions. A Cross-Encoder feeds both strings simultaneously into transformer self-attention layers.
- **Model**: `BAAI/bge-reranker-large` (560M parameters) or `microsoft/mdeberta-v3-base` (278M parameters).
- **Format**: `[CLS] s1_name, s1_addr, s1_country [SEP] cand_name, cand_addr [SEP]`.
- **Inference Optimization**: Exported to TensorRT FP16 execution engines. Processes ~1,500 pairs/sec per RTX 4090.
- **Expected Precision**: $\mathbf{\ge 99.0\%}$.

### Phase 3: Global Bipartite Graph Assignment
Standard thresholding evaluates candidate pairs in isolation. In reality, entity resolution across structured sources is a 1-to-many or 1-to-1 bipartite matching constraint.
- Construct bipartite graph $G = (V_{S1}, V_{\text{Target}}, E)$ where edge weights $w(u, v) = \text{prob}(u, v)$.
- Solve using the Jonker-Volgenant or Hungarian algorithm with an edge threshold cutoff.
- Eliminates duplicate assignment collisions in the same source.

---

## 4. Execution Budget & Throughput on RTX 4090

| Pipeline Stage | Data Volume | Batch Size | Throughput | Wall-Clock Time |
| :--- | :--- | :--- | :--- | :--- |
| **1. Target Offline Embedding** | 5,000,000 Target Records | 512 (FP16) | ~1,200 entities/sec | ~1.15 hours |
| **2. Primary Query Embedding** | 2,200,000 Primary Records | 512 (FP16) | ~1,200 entities/sec | ~30 minutes |
| **3. GPU FAISS HNSW Search** | 2,200,000 Queries x 5M Index | 4,096 | ~35,000 queries/sec | ~1.1 minutes |
| **4. Sparse BM25 Search** | 2,200,000 Queries | 16 threads | ~8,000 queries/sec | ~4.5 minutes |
| **5. Cross-Encoder Reranking** | 44,000,000 Candidate Pairs | 256 (TensorRT) | ~3,500 pairs/sec | ~3.5 hours |
| **6. Bipartite Linkage Assignment** | 2,200,000 Subgraphs | 16 threads | ~50,000 entities/sec | ~45 seconds |
| **Total Pipeline Run Time** | - | - | - | **~5.2 hours** |

---

## 5. Software Stack & Python Dependencies

```text
# Accelerated Core
torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
faiss-gpu-cu12>=1.8.0
transformers>=4.40.0
sentence-transformers>=3.0.0
tensorrt>=10.0.0

# Out-of-Core Data Engine
polars>=1.0.0
duckdb>=1.0.0
scipy>=1.12.0
tantivy>=0.21.0
```

---

## 6. Mathematical Score Target

Under this unconstrained architecture:
- Candidate Recall $R \ge 98.2\%$
- Classifier Precision $P \ge 98.8\%$

$$\text{Projected Macro } F_{0.5} = \frac{1.25 \times 0.988 \times 0.982}{0.25 \times 0.988 + 0.982} = \frac{1.2127}{1.229} \approx \mathbf{0.9867}$$

This achieves the score band of the competition winners (**0.986+**).
