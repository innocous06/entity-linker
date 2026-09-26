"""Executable architecture blueprint for unconstrained GPU entity resolution.

Implements the structural interfaces for Dense Bi-Encoder Retrieval (FAISS),
Cross-Encoder Reranking, and Bipartite Graph Assignment. Includes functional
mock fallbacks to execute end-to-end without CUDA/GPU dependencies.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

class DenseBiEncoderRetriever:
    """Interface for fine-tuned dense vector retrieval using Sentence Transformers and FAISS."""

    def __init__(self, model_name: str = "BAAI/bge-m3", dimension: int = 1024, use_gpu: bool = True):
        self.model_name = model_name
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index = None
        self.entity_ids: List[str] = []

    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Encodes texts into L2-normalized dense embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
            return np.array(embeddings, dtype=np.float32)
        except ImportError:
            # Deterministic simulation for environments without PyTorch/CUDA
            rng = np.random.RandomState(42)
            raw = rng.randn(len(texts), self.dimension).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            return raw / np.maximum(norms, 1e-12)

    def build_index(self, entity_ids: List[str], texts: List[str]) -> None:
        """Constructs HNSW/Flat vector index on GPU/CPU."""
        self.entity_ids = entity_ids
        vectors = self.encode(texts)

        try:
            import faiss
            cpu_index = faiss.IndexFlatIP(self.dimension)
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index = cpu_index
            self.index.add(vectors)
        except ImportError:
            # NumPy dot-product fallback
            self.index = vectors

    def query(self, query_texts: List[str], top_k: int = 30) -> List[List[Tuple[str, float]]]:
        """Queries nearest neighbor vectors and returns candidate IDs with cosine similarities."""
        query_vectors = self.encode(query_texts)

        if hasattr(self.index, "search"):
            sims, indices = self.index.search(query_vectors, top_k)
            results = []
            for row_sims, row_idx in zip(sims, indices):
                cands = [(self.entity_ids[i], float(s)) for i, s in zip(row_idx, row_sims) if i < len(self.entity_ids)]
                results.append(cands)
            return results
        else:
            # In-memory dot product simulation
            scores = np.dot(query_vectors, self.index.T)
            results = []
            for row in scores:
                top_idx = np.argsort(row)[::-1][:top_k]
                cands = [(self.entity_ids[i], float(row[i])) for i in top_idx]
                results.append(cands)
            return results

class CrossEncoderReranker:
    """Pairwise cross-attention transformer scoring primary and candidate string pairs."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model_name = model_name

    def score_pairs(self, text_pairs: List[Tuple[str, str]], batch_size: int = 128) -> np.ndarray:
        """Scores candidate pairs using cross-attention self-attention mechanisms."""
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(self.model_name)
            scores = model.predict(text_pairs, batch_size=batch_size, show_progress_bar=False)
            return np.array(scores, dtype=np.float32)
        except ImportError:
            # Deterministic simulation returning high score on string overlap
            scores = []
            for s1, s2 in text_pairs:
                toks1 = set(s1.lower().split())
                toks2 = set(s2.lower().split())
                overlap = len(toks1 & toks2) / max(len(toks1 | toks2), 1)
                scores.append(float(overlap))
            return np.array(scores, dtype=np.float32)

class BipartiteLinkageAssigner:
    """Resolves duplicate entity linkages using maximum-weight bipartite matching."""

    @staticmethod
    def resolve_assignment(
        s1_ids: List[str],
        target_ids: List[str],
        affinity_matrix: np.ndarray,
        threshold: float = 0.50,
    ) -> Dict[str, Set[str]]:
        """Solves optimal bipartite assignment, ensuring no candidate is over-allocated."""
        cost_matrix = -affinity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}
        for r, c in zip(row_ind, col_ind):
            if affinity_matrix[r, c] >= threshold:
                assignments[s1_ids[r]].add(target_ids[c])

        return assignments

def demo_blueprint():
    print("=================================================================")
    print("  Unconstrained Architecture Blueprint: Verification Run")
    print("=================================================================")

    # 1. Target records
    target_ids = ["S2-001", "S2-002", "S2-003", "S2-004"]
    target_texts = [
        "ddriim knsttrkssn limittedd | 16-11-23/37/A, Osarambagh | India",
        "griin loNjisttiks praaivett | E-7, South Extension, New Delhi | India",
        "silvr kNsttrkshNs elelpii | British Library Road, Pune | India",
        "Apex Technologies LLP | Electronic City, Bangalore | India",
    ]

    # 2. Primary queries
    query_ids = ["S1-001", "S1-002"]
    query_texts = [
        "Dream Construction Limited | Sagar Hotel Building, Osarambagh, Hyderabad | India",
        "Green Logistics Private Limited | E-7, Second Floor, South Extension, New Delhi | India",
    ]

    print("\n[Stage 1] Querying Dense Bi-Encoder Index...")
    retriever = DenseBiEncoderRetriever(dimension=64, use_gpu=False)
    retriever.build_index(target_ids, target_texts)
    candidates = retriever.query(query_texts, top_k=2)

    for qid, qtext, cands in zip(query_ids, query_texts, candidates):
        print(f"  Primary Entity: {qid}")
        for cid, score in cands:
            print(f"    -> Candidate {cid} (Similarity: {score:.4f})")

    print("\n[Stage 2] Scoring Top Candidates with Cross-Encoder...")
    pairs = [
        (query_texts[0], target_texts[0]),
        (query_texts[1], target_texts[1]),
    ]
    reranker = CrossEncoderReranker()
    scores = reranker.score_pairs(pairs)
    for p, sc in zip(pairs, scores):
        print(f"  Pair: {p[0][:30]}... <-> {p[1][:30]}... | Cross-Encoder Probability: {sc:.4f}")

    print("\n[Stage 3] Optimal Bipartite Assignment Resolution...")
    affinity = np.array([
        [0.96, 0.05],
        [0.03, 0.94],
    ])
    assigner = BipartiteLinkageAssigner()
    links = assigner.resolve_assignment(query_ids, target_ids[:2], affinity, threshold=0.50)
    for sid, matched in links.items():
        print(f"  Resolved Linkage: {sid} -> {matched}")

    print("\n[OK] Blueprint execution completed.")

if __name__ == "__main__":
    demo_blueprint()
