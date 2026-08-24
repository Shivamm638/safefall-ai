"""
SafeFall AI - per-class decision calibration.

The bottleneck is Normal Activity: precision 0.71 but recall 0.40, so the model
*can* recognise it and simply loses the argmax to neighbouring classes. Plain
argmax implicitly assumes every class is equally likely and equally costly,
which is false here - the classes are imbalanced and a missed fall costs far
more than a mislabelled bend.

The fix is a per-class multiplier applied to the probability vector before the
argmax. This changes only the decision rule, not the network, so it costs
nothing at inference and cannot overfit the way retraining can.

Weights are found by coordinate ascent on the VALIDATION split, maximising
macro F1 with a floor on fall recall so calibration can never buy accuracy by
quietly missing falls. TEST is scored once at the end.

Run with:  python -m src.calibrate
"""

from __future__ import annotations

import json

import numpy as np

from . import config
from .data import load_split, raw_landmarks

# Calibration may not drop fall recall below this, whatever it does for accuracy.
MIN_FALL_RECALL = 0.92


def objective(y_true: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> float:
    from sklearn.metrics import f1_score, recall_score

    pred = (probs * weights).argmax(1)
    fall_recall = recall_score(y_true, pred, labels=[config.FALL_INDEX],
                               average="micro", zero_division=0)
    if fall_recall < MIN_FALL_RECALL:
        return -1.0
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def search(y_true: np.ndarray, probs: np.ndarray, rounds: int = 6) -> np.ndarray:
    """Coordinate ascent over per-class multipliers."""
    weights = np.ones(config.NUM_CLASSES, dtype=np.float64)
    best = objective(y_true, probs, weights)
    grid = np.concatenate([np.linspace(0.5, 1.0, 11), np.linspace(1.1, 3.0, 20)])

    for _ in range(rounds):
        improved = False
        for c in range(config.NUM_CLASSES):
            original = weights[c]
            for candidate in grid:
                weights[c] = candidate
                value = objective(y_true, probs, weights)
                if value > best + 1e-6:
                    best, original, improved = value, candidate, True
            weights[c] = original
        if not improved:
            break
    return weights


def main() -> None:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    from .inference import SafeFallPredictor

    predictor = SafeFallPredictor()
    print(f"Scoring with: {predictor.engine_name}")

    def probs_for(split_csv):
        df = load_split(split_csv)
        raw = raw_landmarks(df)
        p = np.stack([predictor.predict_landmarks(lm) for lm in raw])
        return p, df["label_index"].to_numpy(dtype=np.int64)

    val_probs, y_val = probs_for(config.VAL_CSV)
    test_probs, y_test = probs_for(config.TEST_CSV)
    predictor.close()

    weights = search(y_val, val_probs)
    print("\nCalibration weights found on validation")
    for name, w in zip(config.CLASS_NAMES, weights):
        print(f"  {name:<18} x{w:.2f}")

    def report(y, probs, w, label):
        pred = (probs * w).argmax(1)
        row = {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "fall_recall": float(recall_score(y, pred, labels=[config.FALL_INDEX],
                                              average="micro", zero_division=0)),
            "normal_recall": float(recall_score(
                y, pred, labels=[config.CLASS_TO_INDEX["Normal Activity"]],
                average="micro", zero_division=0)),
        }
        print(f"  {label:<14}accuracy {row['accuracy']:.4f}   macro F1 {row['macro_f1']:.4f}"
              f"   fall recall {row['fall_recall']:.4f}"
              f"   normal recall {row['normal_recall']:.4f}")
        return row

    ones = np.ones(config.NUM_CLASSES)
    print("\nVALIDATION")
    report(y_val, val_probs, ones, "uncalibrated")
    report(y_val, val_probs, weights, "calibrated")

    print("\nTEST - scored once")
    before = report(y_test, test_probs, ones, "uncalibrated")
    after = report(y_test, test_probs, weights, "calibrated")

    improved = (after["macro_f1"] > before["macro_f1"]
                and after["fall_recall"] >= MIN_FALL_RECALL)
    print(f"\n  macro F1 {before['macro_f1']:.4f} -> {after['macro_f1']:.4f} "
          f"({after['macro_f1'] - before['macro_f1']:+.4f})")
    print(f"  VERDICT: {'ADOPT' if improved else 'REJECT - no real gain'}")

    (config.RESULTS_DIR / "calibration.json").write_text(
        json.dumps({
            "weights": weights.tolist(),
            "class_names": config.CLASS_NAMES,
            "min_fall_recall_constraint": MIN_FALL_RECALL,
            "test_before": before,
            "test_after": after,
            "adopted": bool(improved),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten -> {config.RESULTS_DIR / 'calibration.json'}")


if __name__ == "__main__":
    main()
