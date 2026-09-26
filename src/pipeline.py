"""End-to-end execution pipeline for multilingual entity resolution."""

import argparse
import csv
import gc
from pathlib import Path
from typing import Dict, List, Set, Tuple
import duckdb
import numpy as np
import polars as pl

from src.blocking import InvertedIndexBlocker
from src.features import extract_features_matrix
from src.model import EntityResolutionModel, compute_macro_f05

def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """Reads TSV ground truth file into dictionary mapping s1_id -> set of matched target IDs."""
    gt: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            sid = row[0].strip()
            cands = set(x.strip() for x in row[1].split(",") if x.strip()) if len(row) > 1 else set()
            gt[sid] = cands
    return gt

class EntityResolutionPipeline:
    """Manages normalization, candidate blocking, feature extraction, and model inference."""

    def __init__(self, data_dir: Path, work_dir: Path):
        self.data_dir = data_dir
        self.work_dir = work_dir
        self.cache_dir = work_dir / "cache"
        self.output_dir = work_dir / "output"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.blocker = InvertedIndexBlocker()
        self.model = EntityResolutionModel()

    def run_blocking(self, split: str = "train") -> Path:
        """Executes candidate generation for all countries and target sources."""
        print(f"\n--- Running Blocking for {split.upper()} ---")
        s1_df = pl.read_csv(self.data_dir / f"{split}_source1.tsv", separator="\t", truncate_ragged_lines=True)
        s2_df = pl.read_csv(self.data_dir / f"{split}_source2.tsv", separator="\t", truncate_ragged_lines=True)
        s3_df = pl.read_csv(self.data_dir / f"{split}_source3.tsv", separator="\t", truncate_ragged_lines=True)

        countries = s1_df["country"].unique().to_list()
        partition_files = []

        for country in countries:
            for sx_label, sx_df in [("source2", s2_df), ("source3", s3_df)]:
                out_p = self.cache_dir / f"cand_{split}_{country}_{sx_label}.parquet"
                self.blocker.block_partition(s1_df, sx_df, country, sx_label, out_p, self.cache_dir)
                partition_files.append(out_p)

        del s1_df, s2_df, s3_df
        gc.collect()

        merged_cand_path = self.cache_dir / f"{split}_candidates.parquet"
        con = duckdb.connect()
        con.execute("PRAGMA threads=2;")
        con.execute("PRAGMA preserve_insertion_order=false;")
        f_list = ", ".join([f"'{p.as_posix()}'" for p in partition_files if p.exists()])
        con.execute(f"COPY (SELECT * FROM read_parquet([{f_list}])) TO '{merged_cand_path.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
        total_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{merged_cand_path.as_posix()}')").fetchone()[0]
        con.close()

        print(f"[OK] Merged {total_count:,} candidate pairs into {merged_cand_path.name}")
        return merged_cand_path

    def train_and_evaluate(self) -> Dict[str, float]:
        """Runs blocking, feature extraction, training, and validation scoring."""
        cand_path = self.run_blocking("train")
        gt = load_ground_truth(self.data_dir / "train_ground_truth.tsv")

        s1_df = pl.read_csv(self.data_dir / "train_source1.tsv", separator="\t", truncate_ragged_lines=True)
        s2_df = pl.read_csv(self.data_dir / "train_source2.tsv", separator="\t", truncate_ragged_lines=True)
        s3_df = pl.read_csv(self.data_dir / "train_source3.tsv", separator="\t", truncate_ragged_lines=True)

        s1_lookup = {r["entity_id"]: (r["business_name"], r["business_address"]) for r in s1_df.iter_rows(named=True)}
        target_lookup = {r["entity_id"]: (r["business_name"], r["business_address"]) for r in s2_df.iter_rows(named=True)}
        target_lookup.update({r["entity_id"]: (r["business_name"], r["business_address"]) for r in s3_df.iter_rows(named=True)})

        cand_df = pl.read_parquet(cand_path)
        print(f"Extracting features for {len(cand_df):,} candidate pairs...")
        X = extract_features_matrix(cand_df, s1_lookup, target_lookup)

        s1_ids = cand_df["s1_id"].to_list()
        cand_ids = cand_df["cand_id"].to_list()

        # Labels: 1 if candidate is in ground truth set for s1_id, else 0
        y = np.array([1 if cid in gt.get(sid, set()) else 0 for sid, cid in zip(s1_ids, cand_ids)], dtype=np.int32)
        groups = np.array(s1_ids)

        positives = int(np.sum(y))
        print(f"Dataset summary: {len(y):,} pairs, {positives:,} positive matches ({positives/len(y)*100:.2f}%)")

        metrics = self.model.fit(
            X=X,
            y=y,
            groups=groups,
            val_ground_truth=gt,
            val_s1_ids=s1_ids,
            val_cand_ids=cand_ids,
        )

        print("\n--- Validation Results ---")
        print(f"Optimal Threshold: {metrics['threshold']:.2f}")
        print(f"Validation Precision: {metrics['val_precision']:.4f}")
        print(f"Validation Recall:    {metrics['val_recall']:.4f}")
        print(f"Validation Macro F0.5: {metrics['val_f05']:.4f}")

        self.model.save(self.work_dir / "model.pkl")
        return metrics

    def predict_test(self) -> Path:
        """Generates matching_results.tsv and candidate_pairs.tsv for test split."""
        cand_path = self.run_blocking("test")

        s1_df = pl.read_csv(self.data_dir / "test_source1.tsv", separator="\t", truncate_ragged_lines=True)
        s2_df = pl.read_csv(self.data_dir / "test_source2.tsv", separator="\t", truncate_ragged_lines=True)
        s3_df = pl.read_csv(self.data_dir / "test_source3.tsv", separator="\t", truncate_ragged_lines=True)

        s1_lookup = {r["entity_id"]: (r["business_name"], r["business_address"]) for r in s1_df.iter_rows(named=True)}
        target_lookup = {r["entity_id"]: (r["business_name"], r["business_address"]) for r in s2_df.iter_rows(named=True)}
        target_lookup.update({r["entity_id"]: (r["business_name"], r["business_address"]) for r in s3_df.iter_rows(named=True)})

        cand_df = pl.read_parquet(cand_path)
        print(f"Extracting features for {len(cand_df):,} test candidate pairs...")
        X = extract_features_matrix(cand_df, s1_lookup, target_lookup)

        s1_ids = cand_df["s1_id"].to_list()
        cand_ids = cand_df["cand_id"].to_list()

        predictions = self.model.predict(X, s1_ids, cand_ids)

        # Write matching_results.tsv
        out_matches = self.output_dir / "matching_results.tsv"
        all_test_s1 = s1_df["entity_id"].to_list()

        with open(out_matches, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writerow(["entity_id", "matched_entities"])
            for sid in all_test_s1:
                matches = sorted(list(predictions.get(sid, set())))
                w.writerow([sid, ",".join(matches)])

        # Write candidate_pairs.tsv
        out_cands = self.output_dir / "candidate_pairs.tsv"
        cand_df.select(["s1_id", "cand_id"]).write_csv(out_cands, separator="\t", quote_style="never")

        print(f"[OK] Generated {out_matches.name} ({len(all_test_s1):,} rows)")
        print(f"[OK] Generated {out_cands.name} ({len(cand_df):,} rows)")
        return out_matches

def main():
    parser = argparse.ArgumentParser(description="Multilingual Entity Resolution Pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("sample_data"), help="Directory containing dataset TSVs")
    parser.add_argument("--work-dir", type=Path, default=Path("workspace"), help="Directory for caches, models, outputs")
    parser.add_argument("--train", action="store_true", help="Run training and validation evaluation")
    parser.add_argument("--predict", action="store_true", help="Run inference on test data")
    args = parser.parse_args()

    pipeline = EntityResolutionPipeline(args.data_dir, args.work_dir)
    if args.train:
        pipeline.train_and_evaluate()
    if args.predict:
        pipeline.predict_test()

if __name__ == "__main__":
    main()
