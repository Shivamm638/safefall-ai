"""
SafeFall AI - would more training frames help?

Why this exists
---------------
The dataset is built at ``FRAME_STRIDE = 3``: every third frame of every video.
Rebuilding it at stride 1 would roughly triple the real training frames, and
would take Normal Activity - the class holding macro F1 down, with only 676
training frames - to around two thousand. That is an hour of recomputation, so
it is worth an hour only if more frames would actually help.

Pose-space augmentation already failed to help (4 copies against 6 made no
difference), but that is not the same question. Augmented copies are synthetic
jitter around frames the model has already seen; real frames carry variation
that jitter cannot invent.

Method
------
Answer it by measuring the slope instead of guessing at it. Subsample the
existing training frames *within each video* to simulate coarser strides - every
4th frame of a stride-3 set is a stride-12 set - and train the same
configuration on each. Validation frames are never subsampled, so every point is
scored on identical data.

If validation accuracy has flattened by the time it reaches the full set, the
extra frames a stride-1 rebuild would buy are already redundant and the rebuild
is not worth running. If it is still climbing, it is.

Run with:  python -m src.data_scaling_eval
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from . import config
from .augment import build_augmented_set
from .data import (
    FeatureScaler,
    balance_by_class,
    frame_to_arrays,
    load_split,
    raw_landmarks,
)
from .hparam_search import EPOCHS, train_eval

RESULT_PATH = config.RESULTS_DIR / "data_scaling_eval.json"

# Keep every Nth sampled frame of each video. The dataset is already stride 3,
# so these simulate strides of 12, 6, 4.5 and 3.
KEEP_EVERY = [4, 2, 1]
SEEDS = [config.RANDOM_SEED, 7]


def subsample_by_video(df: pd.DataFrame, keep_every: int) -> pd.DataFrame:
    """Keep every ``keep_every``-th frame within each video, in frame order.

    Subsampling within videos rather than at random is what makes this a stride
    experiment: it thins the temporal sampling exactly as a larger stride would,
    instead of removing whole scenes.
    """
    if keep_every <= 1:
        return df
    parts = []
    for _, group in df.sort_values(["video_id", "frame"]).groupby("video_id", sort=False):
        parts.append(group.iloc[::keep_every])
    return pd.concat(parts).sort_index()


def main() -> None:
    train_df = load_split(config.TRAIN_CSV)
    val_df = load_split(config.VAL_CSV)
    scaler = FeatureScaler.load()

    x_lm_va, x_geo_va, y_va = frame_to_arrays(val_df)
    x_lm_va, x_geo_va = scaler.transform(x_lm_va, x_geo_va)
    val = ({"landmarks": x_lm_va, "geometry": x_geo_va}, y_va)

    cfg = {
        "learning_rate": config.LEARNING_RATE, "l2": config.L2_REGULARISATION,
        "dropout_geo": 0.30, "dropout_head1": 0.45, "dropout_head2": 0.35,
        "width": 1.0, "batch_size": config.BATCH_SIZE,
        "augment_copies": config.AUGMENT_COPIES,
    }

    print(f"Validation frames (never subsampled): {len(val_df):,}")
    print(f"Training the deployed configuration at {len(KEEP_EVERY)} data sizes, "
          f"{len(SEEDS)} seeds each\n")

    results: List[Dict] = []
    for keep in KEEP_EVERY:
        subset = subsample_by_video(train_df, keep)
        balanced = balance_by_class(subset)
        raw = raw_landmarks(balanced)
        y_raw = balanced["label_index"].to_numpy(dtype=np.int64)
        x_lm, x_geo, y = build_augmented_set(raw, y_raw, copies=cfg["augment_copies"])
        x_lm, x_geo = scaler.transform(x_lm, x_geo)
        data = {"train": {cfg["augment_copies"]: (x_lm, x_geo, y)}, "val": val}

        normal = int((balanced["activity"] == "Normal Activity").sum())
        accs, f1s = [], []
        t0 = time.time()
        for seed in SEEDS:
            acc, macro_f1, _ = train_eval(cfg, data, seed=seed, epochs=EPOCHS)
            accs.append(acc)
            f1s.append(macro_f1)

        row = {
            "keep_every": keep,
            "effective_stride": config.FRAME_STRIDE * keep,
            "train_frames": int(len(subset)),
            "normal_activity_frames": normal,
            "augmented_samples": int(len(y)),
            "val_accuracy_mean": float(np.mean(accs)),
            "val_accuracy_std": float(np.std(accs)),
            "val_macro_f1_mean": float(np.mean(f1s)),
            "seeds": SEEDS,
        }
        results.append(row)
        print(f"  stride {row['effective_stride']:>2}  "
              f"{row['train_frames']:>6,} frames "
              f"({normal:>4} Normal)  ->  "
              f"val acc {row['val_accuracy_mean']:.4f} +/- {row['val_accuracy_std']:.4f}  "
              f"macroF1 {row['val_macro_f1_mean']:.4f}   ({time.time() - t0:.0f}s)")

    results.sort(key=lambda r: r["train_frames"])
    first, last = results[0], results[-1]
    gain = last["val_accuracy_mean"] - first["val_accuracy_mean"]
    # Slope over the final doubling is what predicts the next one.
    prev = results[-2]
    recent = last["val_accuracy_mean"] - prev["val_accuracy_mean"]
    noise = max(last["val_accuracy_std"], prev["val_accuracy_std"])

    print(f"\n  {first['train_frames']:,} -> {last['train_frames']:,} frames: "
          f"{gain:+.4f} validation accuracy")
    print(f"  last doubling ({prev['train_frames']:,} -> {last['train_frames']:,}): "
          f"{recent:+.4f}, against seed noise of {noise:.4f}")

    worth_it = recent > noise
    verdict = (
        "Still climbing faster than seed noise - a stride-1 rebuild is likely to help."
        if worth_it else
        "Flat within seed noise - the extra frames a stride-1 rebuild would add are "
        "largely redundant, so the rebuild is not worth running."
    )
    print(f"\n  VERDICT: {verdict}")

    RESULT_PATH.write_text(json.dumps({
        "question": "Would rebuilding the dataset at FRAME_STRIDE=1 improve accuracy?",
        "method": (
            "Subsample training frames within each video to simulate coarser "
            "strides, train the deployed configuration on each, score on the "
            "full untouched validation split. Two seeds per point so the trend "
            "can be read against seed noise."
        ),
        "epochs": EPOCHS,
        "points": results,
        "gain_over_range": gain,
        "gain_last_doubling": recent,
        "seed_noise": noise,
        "rebuild_worth_running": bool(worth_it),
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten -> {RESULT_PATH}")


if __name__ == "__main__":
    main()
