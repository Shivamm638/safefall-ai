"""
SafeFall AI - train the upper-body fall detector.

Why this replaces a threshold
-----------------------------
The full-body CNN reads the kinematic chain down to the ankles, so it cannot be
asked anything when the legs are out of shot - MediaPipe still reports the
missing joints, at invented positions. The first cover for that case was a
single hand-tuned cue, ``shoulder_tilt >= 26 degrees``, which measured 78.5%
accuracy on held-out crops with a 32% false-alarm rate: one upright frame in
three raised an alert.

Two things were being left on the table.

*Hips.* The old detector cropped to head and shoulders only. But the framing
check also rejects frames where the hips ARE in shot and merely the legs are
not - the ordinary webcam view of somebody at a desk. Shoulders plus hips give
the **trunk angle**, which is the canonical fall cue and needs no legs at all.

*Learning.* One threshold on one cue cannot combine evidence. Eleven
visibility-aware features and a small trained classifier can.

Method
------
Sample labelled frames from the video-level splits, and for each one build two
crops that mimic real camera views: a *torso* view (head to hips, legs cut off)
and a *head-and-shoulders* view. Re-run pose estimation on each crop, so the
landmark normalisation is exactly what it would be at inference, and extract
``upper_body_features``. Train on TRAIN, choose the model on VALIDATION, and
report once on TEST.

Run with:  python -m src.upper_body_train
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .pose_utils import UPPER_FEATURE_NAMES, PoseEstimator, upper_body_features

MODEL_PATH = config.MODELS_DIR / "upper_body_model.json"
EVAL_PATH = config.RESULTS_DIR / "upper_body_eval.json"

_WORKER_POSE: Optional[PoseEstimator] = None


# --------------------------------------------------------------------------- #
# Crops that mimic what a real camera sees
# --------------------------------------------------------------------------- #
def _crop(img, cx: float, cy: float, half_w: float, aspect_h_over_w: float):
    """Cut a box around (cx, cy) and resize back to the source resolution.

    Resizing back matters: MediaPipe normalises coordinates by the frame it is
    given, so a crop rescaled to the original size reproduces the coordinate
    system the model will actually meet at inference.
    """
    import cv2

    height, width = img.shape[:2]
    half_h = half_w * aspect_h_over_w
    x0, x1 = int(max(cx - half_w, 0)), int(min(cx + half_w, width))
    y0, y1 = int(max(cy - half_h, 0)), int(min(cy + half_h, height))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 48 or crop.shape[1] < 48:
        return None
    return cv2.resize(crop, (width, height))


def torso_view(img, lm):
    """Head to hips, legs deliberately outside the frame."""
    height, width = img.shape[:2]
    pts = lm[[0, 11, 12, 23, 24], :2] * np.array([width, height])
    cx = float(pts[:, 0].mean())
    top = float(pts[:, 1].min())
    hips = float(lm[[23, 24], 1].mean() * height)
    # Bottom edge just below the hips, so the legs are genuinely absent.
    bottom = hips + (hips - top) * 0.12
    cy = (top + bottom) / 2.0
    half_h = max((bottom - top) / 2.0 * 1.15, 40.0)
    return _crop(img, cx, cy, half_h * width / height, height / width)


def head_shoulders_view(img, lm, pad: float = 1.7):
    """Head and shoulders only - the tightest view the app must still answer."""
    height, width = img.shape[:2]
    pts = lm[[0, 11, 12], :2] * np.array([width, height])
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    half_w = max(float(np.ptp(pts[:, 0])), 40.0) * pad
    return _crop(img, cx, cy, half_w, height / width)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _init_worker() -> None:
    global _WORKER_POSE
    os.environ.setdefault("GLOG_minloglevel", "3")
    _WORKER_POSE = PoseEstimator(static_image_mode=True)


def process_video(task: Dict) -> List[Dict]:
    """Build crop features for every sampled frame of one video."""
    import cv2

    global _WORKER_POSE
    if _WORKER_POSE is None:
        _init_worker()
    pose = _WORKER_POSE

    rows: List[Dict] = []
    cap = cv2.VideoCapture(task["video_path"])
    if not cap.isOpened():
        return rows

    for frame_no, is_fall in task["frames"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_no) - 1, 0))
        ok, frame = cap.read()
        if not ok:
            continue
        full_lm = pose.extract(frame)
        if full_lm is None:
            continue

        for view_name, builder in (("torso", torso_view),
                                   ("head_shoulders", head_shoulders_view)):
            try:
                crop = builder(frame, full_lm)
            except Exception:
                crop = None
            if crop is None:
                continue
            crop_lm = pose.extract(crop)
            if crop_lm is None:
                continue
            height, width = crop.shape[:2]
            features = upper_body_features(crop_lm, aspect=width / max(height, 1))
            if not np.all(np.isfinite(features)):
                continue
            row = {name: float(v) for name, v in zip(UPPER_FEATURE_NAMES, features)}
            row.update({"is_fall": int(is_fall), "view": view_name,
                        "video_id": task["video_id"]})
            rows.append(row)

    cap.release()
    return rows


def build_split(split_csv, lookup: Dict, limit: int, workers: int,
                label: str) -> pd.DataFrame:
    """Balanced sample of one split, turned into upper-body crop features."""
    df = pd.read_csv(split_csv, usecols=["video_id", "frame", "activity"])
    rng = np.random.RandomState(config.RANDOM_SEED)

    falls = df[df["activity"] == config.FALL_CLASS]
    others = df[df["activity"] != config.FALL_CLASS]
    take = min(limit // 2, len(falls), len(others))
    picked = pd.concat([
        falls.iloc[rng.choice(len(falls), take, replace=False)],
        others.iloc[rng.choice(len(others), take, replace=False)],
    ])
    picked = picked.assign(is_fall=(picked["activity"] == config.FALL_CLASS).astype(int))

    tasks = []
    for video_id, group in picked.groupby("video_id"):
        path = lookup.get(video_id)
        if not path:
            continue
        frames = sorted(zip(group["frame"].tolist(), group["is_fall"].tolist()))
        tasks.append({"video_id": video_id, "video_path": path, "frames": frames})

    print(f"  {label}: {len(picked):,} frames across {len(tasks)} videos", flush=True)
    rows: List[Dict] = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        for i, out in enumerate(pool.imap_unordered(process_video, tasks, chunksize=1), 1):
            rows.extend(out)
            if i % 10 == 0 or i == len(tasks):
                print(f"    [{i}/{len(tasks)}] {time.time() - t0:5.1f}s  "
                      f"{len(rows):,} crops", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def evaluate(y_true, prob, threshold: float) -> Dict:
    pred = (prob >= threshold).astype(int)
    positives = max(int((y_true == 0).sum()), 1)
    return {
        "accuracy": float((pred == y_true).mean()),
        "fall_recall": float(pred[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
        "fall_precision": float(y_true[pred == 1].mean()) if (pred == 1).any() else 0.0,
        "false_alarm_rate": float(((pred == 1) & (y_true == 0)).sum() / positives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the upper-body fall detector")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--train-samples", type=int, default=2400)
    parser.add_argument("--eval-samples", type=int, default=800)
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    index = pd.read_csv(config.VIDEO_INDEX_CSV)
    lookup = dict(zip(index["video_id"], index["video_path"]))

    print(f"Building upper-body crops with {args.workers} workers "
          "(reads video, please wait)")
    train = build_split(config.TRAIN_CSV, lookup, args.train_samples, args.workers, "train")
    val = build_split(config.VAL_CSV, lookup, args.eval_samples, args.workers, "val")
    test = build_split(config.TEST_CSV, lookup, args.eval_samples, args.workers, "test")

    for name, frame in (("train", train), ("val", val), ("test", test)):
        if len(frame) < 60:
            raise SystemExit(f"Only {len(frame)} usable {name} crops - cannot conclude.")
        print(f"  {name}: {len(frame):,} crops "
              f"({int(frame.is_fall.sum())} falls, "
              f"{(frame.view == 'torso').sum()} torso / "
              f"{(frame.view == 'head_shoulders').sum()} head-and-shoulders)")

    columns = list(UPPER_FEATURE_NAMES)
    x_train, y_train = train[columns].to_numpy(np.float32), train["is_fall"].to_numpy()
    x_val, y_val = val[columns].to_numpy(np.float32), val["is_fall"].to_numpy()
    x_test, y_test = test[columns].to_numpy(np.float32), test["is_fall"].to_numpy()

    scaler = StandardScaler().fit(x_train)
    x_train_s, x_val_s, x_test_s = (scaler.transform(a) for a in (x_train, x_val, x_test))

    candidates = {
        "logistic": LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced",
                                       random_state=config.RANDOM_SEED),
        "mlp": MLPClassifier(hidden_layer_sizes=(24, 12), max_iter=1200,
                             alpha=1e-3, random_state=config.RANDOM_SEED),
    }

    print("\nVALIDATION - choosing model and operating threshold")
    best = None
    for name, model in candidates.items():
        model.fit(x_train_s, y_train)
        prob = model.predict_proba(x_val_s)[:, 1]
        # Recall matters more than precision for a fall alarm, but the old
        # detector's real failing was false alarms, so score on accuracy while
        # refusing thresholds that drop recall below the previous 0.90.
        for threshold in np.arange(0.20, 0.81, 0.02):
            scores = evaluate(y_val, prob, threshold)
            if scores["fall_recall"] < 0.90:
                continue
            if best is None or scores["accuracy"] > best["scores"]["accuracy"]:
                best = {"name": name, "model": model,
                        "threshold": float(threshold), "scores": scores}
        top = evaluate(y_val, prob, 0.5)
        print(f"  {name:<9} at 0.50: accuracy {top['accuracy']:.3f}  "
              f"recall {top['fall_recall']:.3f}  "
              f"false alarms {top['false_alarm_rate']:.3f}")

    if best is None:
        raise SystemExit("No threshold reached 0.90 recall on validation.")
    print(f"  -> chosen: {best['name']} at threshold {best['threshold']:.2f} "
          f"(validation accuracy {best['scores']['accuracy']:.3f})")

    model = best["model"]
    test_prob = model.predict_proba(x_test_s)[:, 1]
    test_scores = evaluate(y_test, test_prob, best["threshold"])

    print("\nTEST - scored once with the chosen model and threshold")
    for key, value in test_scores.items():
        print(f"  {key:<18} {value:.3f}")

    per_view = {}
    for view in ("torso", "head_shoulders"):
        mask = (test["view"] == view).to_numpy()
        if mask.sum() >= 20:
            per_view[view] = evaluate(y_test[mask], test_prob[mask], best["threshold"])
            print(f"  {view:<16} accuracy {per_view[view]['accuracy']:.3f}  "
                  f"recall {per_view[view]['fall_recall']:.3f}  "
                  f"false alarms {per_view[view]['false_alarm_rate']:.3f}")

    # ------------------------------------------------------------- export ----
    if best["name"] == "logistic":
        weights = {"kind": "logistic",
                   "coef": model.coef_[0].tolist(),
                   "intercept": float(model.intercept_[0])}
    else:
        weights = {"kind": "mlp",
                   "coefs": [c.tolist() for c in model.coefs_],
                   "intercepts": [b.tolist() for b in model.intercepts_],
                   "activation": model.activation,
                   "out_activation": model.out_activation_}

    payload = {
        "features": list(UPPER_FEATURE_NAMES),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": best["threshold"],
        "model": weights,
    }
    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result = {
        "detector": f"learned upper-body ({best['name']})",
        "features": list(UPPER_FEATURE_NAMES),
        "threshold": best["threshold"],
        "validation_accuracy": best["scores"]["accuracy"],
        "test_accuracy": test_scores["accuracy"],
        "test_fall_recall": test_scores["fall_recall"],
        "test_fall_precision": test_scores["fall_precision"],
        "test_false_alarm_rate": test_scores["false_alarm_rate"],
        "test_crops": int(len(test)),
        "per_view": per_view,
        "train_crops": int(len(train)),
        "note": (
            "Measured on crops of held-out Le2i frames that mimic real camera "
            "views: a torso view (head to hips, legs out of shot) and a "
            "head-and-shoulders view. Fall vs not-fall only - the upper body "
            "cannot separate walking from standing."
        ),
    }
    EVAL_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWritten -> {MODEL_PATH}\n         -> {EVAL_PATH}")


if __name__ == "__main__":
    main()
