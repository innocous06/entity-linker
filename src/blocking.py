"""Memory-bounded streaming candidate blocker using IDF-weighted inverted indexes."""

import gc
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import duckdb
import polars as pl
from tqdm.auto import tqdm

from src.normalizer import get_name_tokens, get_address_tokens

# Thresholds tuned to maximize recall while bounding index size
DEFAULT_MAX_NAME_DOCS = 12000
DEFAULT_MAX_ADDR_DOCS = 2000
DEFAULT_MAX_CANDS = 50
DEFAULT_CHUNK_FLUSH = 100_000

class InvertedIndexBlocker:
    """IDF-weighted inverted index blocker with streaming disk-backed chunk flushing."""

    def __init__(
        self,
        max_name_docs: int = DEFAULT_MAX_NAME_DOCS,
        max_addr_docs: int = DEFAULT_MAX_ADDR_DOCS,
        max_candidates: int = DEFAULT_MAX_CANDS,
        chunk_flush_size: int = DEFAULT_CHUNK_FLUSH,
    ):
        self.max_name_docs = max_name_docs
        self.max_addr_docs = max_addr_docs
        self.max_candidates = max_candidates
        self.chunk_flush_size = chunk_flush_size

    def build_index(
        self,
        target_df: pl.DataFrame,
    ) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]], Dict[str, float], Dict[str, float]]:
        """Constructs name and address inverted indexes with on-the-fly stopword pruning."""
        name_idx = defaultdict(set)
        addr_idx = defaultdict(set)
        pruned_name = set()
        pruned_addr = set()

        total_docs = len(target_df)
        records = target_df.select(["entity_id", "business_name", "business_address"]).iter_rows(named=True)

        for row in records:
            cid = row["entity_id"]
            for t in get_name_tokens(row["business_name"]):
                if t not in pruned_name:
                    s = name_idx[t]
                    if len(s) < self.max_name_docs:
                        s.add(cid)
                    else:
                        pruned_name.add(t)
                        del name_idx[t]

            for t in get_address_tokens(row["business_address"]):
                if t not in pruned_addr:
                    s = addr_idx[t]
                    if len(s) < self.max_addr_docs:
                        s.add(cid)
                    else:
                        pruned_addr.add(t)
                        del addr_idx[t]

        # Compute smoothed IDF weights: log((N + 1) / (df + 1))
        name_idf = {t: math.log((total_docs + 1) / (len(v) + 1)) * 3.0 for t, v in name_idx.items()}
        addr_idf = {t: math.log((total_docs + 1) / (len(v) + 1)) * 2.0 for t, v in addr_idx.items()}

        # Freeze sets to tuples for memory compactness
        name_idx_frozen = {k: tuple(v) for k, v in name_idx.items()}
        addr_idx_frozen = {k: tuple(v) for k, v in addr_idx.items()}

        del name_idx, addr_idx, pruned_name, pruned_addr
        gc.collect()

        return name_idx_frozen, addr_idx_frozen, name_idf, addr_idf

    def block_partition(
        self,
        s1_df: pl.DataFrame,
        sx_df: pl.DataFrame,
        country: str,
        source_name: str,
        output_parquet: Path,
        cache_dir: Path,
    ) -> int:
        """Generates candidate pairs for one (country, source) combination."""
        if output_parquet.exists() and output_parquet.stat().st_size > 1000:
            print(f"  [CACHE HIT] {output_parquet.name}")
            return duckdb.connect().execute(f"SELECT COUNT(*) FROM read_parquet('{output_parquet.as_posix()}')").fetchone()[0]

        s1_c = s1_df.filter(pl.col("country") == country)
        sx_c = sx_df.filter(pl.col("country") == country)

        if len(s1_c) == 0 or len(sx_c) == 0:
            pl.DataFrame({
                "s1_id": pl.Series([], dtype=pl.Utf8),
                "cand_id": pl.Series([], dtype=pl.Utf8),
                "blocking_score": pl.Series([], dtype=pl.Float32),
                "source": pl.Series([], dtype=pl.Utf8),
                "country": pl.Series([], dtype=pl.Utf8),
            }).write_parquet(output_parquet)
            return 0

        t0 = time.time()
        name_idx, addr_idx, name_idf, addr_idf = self.build_index(sx_c)
        print(f"  [{country}/{source_name}] Indexed {len(name_idx):,} name tokens, {len(addr_idx):,} addr tokens in {time.time()-t0:.1f}s")

        chunk_files = []
        chunk_idx = 0
        pairs_s1: List[str] = []
        pairs_cx: List[str] = []
        pairs_sc: List[float] = []

        def flush_chunk():
            nonlocal chunk_idx
            if not pairs_s1:
                return
            chk_p = cache_dir / f"tmp_{output_parquet.stem}_{chunk_idx}.parquet"
            pl.DataFrame({
                "s1_id": pairs_s1,
                "cand_id": pairs_cx,
                "blocking_score": pairs_sc,
                "source": [source_name] * len(pairs_s1),
                "country": [country] * len(pairs_s1),
            }).write_parquet(chk_p, compression="snappy")
            chunk_files.append(chk_p)
            chunk_idx += 1
            pairs_s1.clear()
            pairs_cx.clear()
            pairs_sc.clear()

        s1_records = s1_c.select(["entity_id", "business_name", "business_address"]).iter_rows(named=True)
        for row in tqdm(s1_records, total=len(s1_c), desc=f"  Blocking {country}/{source_name}", leave=False):
            sid = row["entity_id"]
            scores: Dict[str, float] = defaultdict(float)

            for t in get_name_tokens(row["business_name"]):
                w = name_idf.get(t, 0.0)
                for cid in name_idx.get(t, ()):
                    scores[cid] += w

            for t in get_address_tokens(row["business_address"]):
                w = addr_idf.get(t, 0.0)
                for cid in addr_idx.get(t, ()):
                    scores[cid] += w

            if not scores:
                continue

            # Rank by accumulated IDF score
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.max_candidates]
            for cid, sc in top:
                pairs_s1.append(sid)
                pairs_cx.append(cid)
                pairs_sc.append(sc)

            if len(pairs_s1) >= self.chunk_flush_size:
                flush_chunk()

        flush_chunk()
        del name_idx, addr_idx, name_idf, addr_idf
        gc.collect()

        # Merge chunk files into single Parquet via DuckDB streaming
        if chunk_files:
            con = duckdb.connect()
            con.execute("PRAGMA threads=2;")
            con.execute("PRAGMA preserve_insertion_order=false;")
            f_list = ", ".join([f"'{p.as_posix()}'" for p in chunk_files])
            con.execute(f"COPY (SELECT * FROM read_parquet([{f_list}])) TO '{output_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
            total_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_parquet.as_posix()}')").fetchone()[0]
            con.close()
            for p in chunk_files:
                try:
                    p.unlink()
                except OSError:
                    pass
        else:
            pl.DataFrame({
                "s1_id": pl.Series([], dtype=pl.Utf8),
                "cand_id": pl.Series([], dtype=pl.Utf8),
                "blocking_score": pl.Series([], dtype=pl.Float32),
                "source": pl.Series([], dtype=pl.Utf8),
                "country": pl.Series([], dtype=pl.Utf8),
            }).write_parquet(output_parquet)
            total_pairs = 0

        print(f"  [{country}/{source_name}] Emitted {total_pairs:,} candidate pairs -> {output_parquet.name}")
        return total_pairs
