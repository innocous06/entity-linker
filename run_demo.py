"""One-click demonstration script running full pipeline on sample data."""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import EntityResolutionPipeline

def main():
    print("=================================================================")
    print("  Multilingual Entity Resolution - End-to-End Demonstration")
    print("=================================================================")

    data_dir = ROOT / "sample_data"
    work_dir = ROOT / "workspace"

    if not (data_dir / "train_source1.tsv").exists():
        raise FileNotFoundError(f"Sample data not found in {data_dir}. Please ensure sample_data TSV files exist.")

    pipeline = EntityResolutionPipeline(data_dir=data_dir, work_dir=work_dir)

    print("\n[STEP 1/2] Training & Validation Evaluation...")
    metrics = pipeline.train_and_evaluate()

    print("\n[STEP 2/2] Generating Test Predictions...")
    pipeline.predict_test()

    print("\n=================================================================")
    print(f"  Demonstration Complete.")
    print(f"  Validation F0.5: {metrics['val_f05']:.4f}")
    print(f"  Outputs saved in: {work_dir / 'output'}")
    print("=================================================================")

if __name__ == "__main__":
    main()
