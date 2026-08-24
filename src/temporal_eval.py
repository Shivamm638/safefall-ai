"""
SafeFall AI - does temporal smoothing help, and by how much?

Single-frame classification throws away the strongest signal a video has: the
previous frame. A resident does not flicker between standing and falling 20
times a second, so averaging the probability vector over a short window should
remove exactly the jitter that a per-frame classifier produces.

That is a claim worth testing rather than assuming. This script applies a
rolling mean of the predicted probabilities **within each video** (frames are
never mixed across recordings) and sweeps the window length.

The window is chosen on VALIDATION and reported once on TEST.

Run with:  python -m src.temporal_eval
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .data import load_split, raw_landmarks


def rolling_probabilities(df: pd.DataFrame, probs: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean of the probability vectors, per video, causal (past only).

    Causal on purpose: a live monitor cannot see the future, so smoothing that
    peeked at later frames would report an accuracy the deployed system could
    never reach.
    """
    if window <= 1:
        return probs

    out = np.empty_like(probs)
    order = df.sort_values(["video_id", "frame"]).index.to_numpy()
    positions = {idx: i for i, idx in enumerate(df.index)}

    video_ids = df["video_id"].to_numpy()
    buffer: list = []
    current = None
    for idx in order:
        i = positions[idx]
        if video_ids[i] != current:
            current = video_ids[i]
            buffer = []
        buffer.append(probs[i])
        if len(buffer) > window:
            buffer.pop(0)
        out[i] = np.mean(buffer, axis=0)
    return out


def main() -> None:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    from .inference import SafeFallPredictor

    predictor = SafeFallPredictor()
    print(f"Scoring with: {predictor.engine_name}")

    def raw_probs(split_csv):
        df = load_split(split_csv)
        raw = raw_landmarks(df)
        probs = np.stack([predictor.predict_landmarks(lm) for lm in raw])
        return df, probs, df["label_index"].to_numpy(dtype=np.int64)

    val_df, val_probs, y_val = raw_probs(config.VAL_CSV)
    test_df, test_probs, y_test = raw_probs(config.TEST_CSV)
    predictor.close()

    def score(df, probs, y, window):
        pred = rolling_probabilities(df, probs, window).argmax(1)
        return {
            "window": window,
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "fall_recall": float(recall_score(y, pred, labels=[config.FALL_INDEX],
                                              average="micro", zero_division=0)),
        }

    windows = [1, 3, 5, 7, 9, 11, 15]
    print("\nVALIDATION - choosing the window")
    print(f"  {'window':>7}{'accuracy':>11}{'macro F1':>11}{'fall recall':>13}")
    val_rows = []
    for w in windows:
        r = score(val_df, val_probs, y_val, w)
        val_rows.append(r)
        print(f"  {w:>7}{r['accuracy']:>11.4f}{r['macro_f1']:>11.4f}{r['fall_recall']:>13.4f}")

    best = max(val_rows, key=lambda r: r["accuracy"])
    print(f"\n  -> chosen window: {best['window']}")

    print("\nTEST - scored once")
    print(f"  {'window':>7}{'accuracy':>11}{'macro F1':>11}{'fall recall':>13}")
    test_rows = []
    for w in windows:
        r = score(test_df, test_probs, y_test, w)
        test_rows.append(r)
        marker = "  <- chosen" if w == best["window"] else ""
        print(f"  {w:>7}{r['accuracy']:>11.4f}{r['macro_f1']:>11.4f}"
              f"{r['fall_recall']:>13.4f}{marker}")

    chosen_test = next(r for r in test_rows if r["window"] == best["window"])
    baseline = next(r for r in test_rows if r["window"] == 1)
    print(f"\n  accuracy    {baseline['accuracy']:.4f} -> {chosen_test['accuracy']:.4f}"
          f"  ({chosen_test['accuracy'] - baseline['accuracy']:+.4f})")
    print(f"  macro F1    {baseline['macro_f1']:.4f} -> {chosen_test['macro_f1']:.4f}"
          f"  ({chosen_test['macro_f1'] - baseline['macro_f1']:+.4f})")
    print(f"  fall recall {baseline['fall_recall']:.4f} -> {chosen_test['fall_recall']:.4f}"
          f"  ({chosen_test['fall_recall'] - baseline['fall_recall']:+.4f})")

    (config.RESULTS_DIR / "temporal_eval.json").write_text(
        json.dumps({
            "chosen_window": best["window"],
            "validation": val_rows,
            "test": test_rows,
            "note": ("Causal rolling mean of probability vectors within each video. "
                     "Applies to video and live monitoring; a single uploaded photo "
                     "has no temporal context and is unaffected."),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten -> {config.RESULTS_DIR / 'temporal_eval.json'}")


if __name__ == "__main__":
    main()
