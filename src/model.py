"""LightGBM classification and threshold calibration for macro F0.5 optimization."""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.features import FEATURE_NAMES

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, float, float]:
    """Computes competition macro-averaged F0.5, Precision, and Recall across all entities."""
    all_s1_ids = list(ground_truth.keys())
    if not all_s1_ids:
        return 0.0, 0.0, 0.0

    precisions = []
    recalls = []
    f05_scores = []

    for sid in all_s1_ids:
        true_set = ground_truth.get(sid, set())
        pred_set = predictions.get(sid, set())

        # Exact match on empty sets (singletons)
        if not true_set and not pred_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f05_scores.append(1.0)
            continue

        if not true_set and pred_set:
            precisions.append(0.0)
            recalls.append(1.0)
            f05_scores.append(0.0)
            continue

        if true_set and not pred_set:
            precisions.append(1.0)
            recalls.append(0.0)
            f05_scores.append(0.0)
            continue

        tp = len(true_set & pred_set)
        p = tp / len(pred_set) if pred_set else 0.0
        r = tp / len(true_set) if true_set else 0.0

        denom = 0.25 * p + r
        f05 = (1.25 * p * r) / denom if denom > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f05)

    return float(np.mean(f05_scores)), float(np.mean(precisions)), float(np.mean(recalls))

class EntityResolutionModel:
    """Gradient boosted decision tree classifier with calibrated F0.5 decision boundary."""

    def __init__(self, model_params: Optional[Dict] = None):
        self.model_params = model_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 2.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.clf: Optional[lgb.LGBMClassifier] = None
        self.optimal_threshold: float = 0.50

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        val_ground_truth: Dict[str, Set[str]],
        val_s1_ids: List[str],
        val_cand_ids: List[str],
    ) -> Dict[str, float]:
        """Trains LightGBM with group-aware cross-validation and threshold optimization."""
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=groups))

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        self.clf = lgb.LGBMClassifier(**self.model_params)
        self.clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        val_probs = self.clf.predict_proba(X_val)[:, 1]

        # Map validation predictions back to entities
        val_s1_arr = np.array(val_s1_ids)[val_idx]
        val_cand_arr = np.array(val_cand_ids)[val_idx]
        val_unique_s1 = set(val_s1_arr)
        scoped_gt = {sid: val_ground_truth[sid] for sid in val_unique_s1 if sid in val_ground_truth}

        best_th = 0.50
        best_f05 = -1.0
        best_p = 0.0
        best_r = 0.0

        for th in np.arange(0.20, 0.85, 0.05):
            preds: Dict[str, Set[str]] = {sid: set() for sid in scoped_gt}
            mask = val_probs >= th
            for sid, cid in zip(val_s1_arr[mask], val_cand_arr[mask]):
                if sid in preds:
                    preds[sid].add(cid)

            f05, p, r = compute_macro_f05(preds, scoped_gt)
            if f05 > best_f05:
                best_f05 = f05
                best_th = float(th)
                best_p = p
                best_r = r

        self.optimal_threshold = best_th
        return {
            "val_f05": best_f05,
            "val_precision": best_p,
            "val_recall": best_r,
            "threshold": best_th,
        }

    def predict(
        self,
        X: np.ndarray,
        s1_ids: List[str],
        cand_ids: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Set[str]]:
        """Generates matched entity sets for each S1 entity using the calibrated threshold."""
        th = threshold if threshold is not None else self.optimal_threshold
        if self.clf is None or len(X) == 0:
            return {sid: set() for sid in s1_ids}

        probs = self.clf.predict_proba(X)[:, 1]
        results: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}

        mask = probs >= th
        for sid, cid in zip(np.array(s1_ids)[mask], np.array(cand_ids)[mask]):
            results[sid].add(cid)

        return results

    def save(self, filepath: Path) -> None:
        """Serializes model and optimal threshold to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"clf": self.clf, "threshold": self.optimal_threshold}, f)

    def load(self, filepath: Path) -> None:
        """Loads serialized model and threshold."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.clf = data["clf"]
            self.optimal_threshold = data["threshold"]
