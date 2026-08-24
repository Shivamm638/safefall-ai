"""
SafeFall AI - measure the upper-body fall detector.

The full-body CNN cannot be used when only head and shoulders are in shot, so a
separate geometric detector covers that case. It would be irresponsible to ship
that detector on intuition: this script gives it a real number.

Method
------
Take labelled frames from the held-out videos, crop each one to a
head-and-shoulders view that mimics a laptop webcam (keeping the frame's aspect
ratio, because normalised landmark coordinates skew otherwise), re-run pose
estimation on the crop, and score the detector against the Le2i ground truth.

Thresholds are chosen on the VALIDATION videos and reported once on TEST.

Run with:  python -m src.upper_body_eval
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .pose_utils import PoseEstimator, upper_body_metrics

SAMPLES_PER_SPLIT = 500


def head_shoulders_crop(img, lm, pad: float = 1.7):
    """Crop to head+shoulders, preserving the source aspect ratio."""
    import cv2

    height, width = img.shape[:2]
    pts = lm[[0, 11, 12], :2] * np.array([width, height])
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    half_w = max(float(np.ptp(pts[:, 0])), 40.0) * pad
    half_h = half_w * height / width

    x0, x1 = int(max(cx - half_w, 0)), int(min(cx + half_w, width))
    y0, y1 = int(max(cy - half_h, 0)), int(min(cy + half_h, height))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 40:
        return None
    return cv2.resize(crop, (width, height))


def collect(split_csv, lookup, pose, limit: int) -> pd.DataFrame:
    """Metrics + ground truth for upper-body crops of one split."""
    import cv2

    df = pd.read_csv(split_csv)
    rng = np.random.RandomState(config.RANDOM_SEED)
    # Balance the sample so the threshold is not tuned on a mostly-negative set.
    falls = df[df["activity"] == config.FALL_CLASS]
    others = df[df["activity"] != config.FALL_CLASS]
    take = min(limit // 2, len(falls), len(others))
    picked = pd.concat([
        falls.iloc[rng.choice(len(falls), take, replace=False)],
        others.iloc[rng.choice(len(others), take, replace=False)],
    ])

    rows = []
    cache = {}
    for _, row in picked.iterrows():
        path = lookup.get(row["video_id"])
        if not path:
            continue
        cap = cache.get(path)
        if cap is None:
            cap = cv2.VideoCapture(path)
            cache[path] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(row["frame"]) - 1, 0))
        ok, frame = cap.read()
        if not ok:
            continue

        full_lm = pose.extract(frame)
        if full_lm is None:
            continue
        crop = head_shoulders_crop(frame, full_lm)
        if crop is None:
            continue
        crop_lm = pose.extract(crop)
        if crop_lm is None:
            continue

        height, width = crop.shape[:2]
        m = upper_body_metrics(crop_lm, aspect=width / max(height, 1))
        rows.append({
            "is_fall": int(row["activity"] == config.FALL_CLASS),
            "head_axis_angle": m["head_axis_angle"],
            "shoulder_tilt": m["shoulder_tilt"],
        })

    for cap in cache.values():
        cap.release()
    return pd.DataFrame(rows)


def sweep(df: pd.DataFrame, column: str):
    """Best single-threshold accuracy for one cue."""
    from sklearn.metrics import accuracy_score, recall_score

    best = None
    for t in np.arange(10, 86, 2.0):
        pred = (df[column] >= t).astype(int)
        acc = accuracy_score(df["is_fall"], pred)
        if best is None or acc > best["accuracy"]:
            best = {
                "cue": column,
                "threshold": float(t),
                "accuracy": float(acc),
                "recall": float(recall_score(df["is_fall"], pred, zero_division=0)),
                "false_alarm_rate": float(
                    ((pred == 1) & (df["is_fall"] == 0)).sum()
                    / max((df["is_fall"] == 0).sum(), 1)
                ),
            }
    return best


def main() -> None:
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    index = pd.read_csv(config.VIDEO_INDEX_CSV)
    lookup = dict(zip(index["video_id"], index["video_path"]))
    pose = PoseEstimator(static_image_mode=True)

    print("Building upper-body crops (this reads video, please wait)...")
    val = collect(config.VAL_CSV, lookup, pose, SAMPLES_PER_SPLIT)
    test = collect(config.TEST_CSV, lookup, pose, SAMPLES_PER_SPLIT)
    pose.close()

    print(f"  validation crops: {len(val):,}  ({int(val.is_fall.sum())} falls)")
    print(f"  test crops      : {len(test):,}  ({int(test.is_fall.sum())} falls)")
    if len(val) < 40 or len(test) < 40:
        raise SystemExit("Not enough usable crops to draw a conclusion.")

    print("\nVALIDATION - choosing the cue and threshold")
    candidates = [sweep(val, "shoulder_tilt"), sweep(val, "head_axis_angle")]
    for c in candidates:
        print(f"  {c['cue']:<18} threshold {c['threshold']:>5.1f}  "
              f"accuracy {c['accuracy']:.3f}  recall {c['recall']:.3f}  "
              f"false alarms {c['false_alarm_rate']:.3f}")
    chosen = max(candidates, key=lambda c: c["accuracy"])
    print(f"  -> chosen: {chosen['cue']} >= {chosen['threshold']:.1f}")

    pred = (test[chosen["cue"]] >= chosen["threshold"]).astype(int)
    result = {
        "cue": chosen["cue"],
        "threshold": chosen["threshold"],
        "validation_accuracy": chosen["accuracy"],
        "test_accuracy": float(accuracy_score(test["is_fall"], pred)),
        "test_fall_recall": float(recall_score(test["is_fall"], pred, zero_division=0)),
        "test_fall_precision": float(precision_score(test["is_fall"], pred, zero_division=0)),
        "test_false_alarm_rate": float(
            ((pred == 1) & (test["is_fall"] == 0)).sum()
            / max((test["is_fall"] == 0).sum(), 1)
        ),
        "test_crops": int(len(test)),
        "note": (
            "Measured on head-and-shoulders crops of held-out Le2i frames, which "
            "simulate a laptop-webcam view. Fall vs not-fall only - the upper body "
            "cannot distinguish walking from standing."
        ),
    }
    print("\nTEST - scored once with the chosen threshold")
    print(f"  accuracy       {result['test_accuracy']:.3f}")
    print(f"  fall recall    {result['test_fall_recall']:.3f}")
    print(f"  fall precision {result['test_fall_precision']:.3f}")
    print(f"  false alarms   {result['test_false_alarm_rate']:.3f} of non-fall crops")

    (config.RESULTS_DIR / "upper_body_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"\nWritten -> {config.RESULTS_DIR / 'upper_body_eval.json'}")


if __name__ == "__main__":
    main()
