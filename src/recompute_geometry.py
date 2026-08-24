"""
SafeFall AI - recompute the geometric feature columns from the stored landmarks.

Pose extraction is the expensive step (about 12 minutes across 17 GB of video),
but the raw MediaPipe landmarks are kept in the feature tables.  Whenever the
feature engineering in ``pose_utils.py`` changes, this script regenerates the
``geo_*`` and motion columns in seconds instead of re-decoding every video.

Run with:  python -m src.recompute_geometry
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import LANDMARK_COLUMNS
from .pose_utils import compute_geometric_features


def recompute(path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    raw = df[LANDMARK_COLUMNS].to_numpy(dtype=np.float32)
    raw = raw.reshape(-1, config.NUM_LANDMARKS, config.LANDMARK_CHANNELS)

    features = [compute_geometric_features(sample) for sample in raw]
    table = pd.DataFrame(features)[config.GEOMETRIC_FEATURE_NAMES]
    for name in config.GEOMETRIC_FEATURE_NAMES:
        df[f"geo_{name}"] = table[name].round(5).to_numpy()

    # Motion features depend on hip_y, so refresh them from the new geometry.
    df = df.sort_values(["video_id", "frame"]).reset_index(drop=True)
    hip_x = ((df["lm23_x"] + df["lm24_x"]) / 2.0).to_numpy()
    hip_y = df["geo_hip_y"].to_numpy()
    aspect = df["geo_bbox_aspect_ratio"].to_numpy()
    frames = df["frame"].to_numpy()
    same_video = df["video_id"].to_numpy()[1:] == df["video_id"].to_numpy()[:-1]

    fps = 25.0
    gap = np.maximum(np.diff(frames), 1) / fps
    vx = np.concatenate([[0.0], np.where(same_video, np.diff(hip_x) / gap, 0.0)])
    vy = np.concatenate([[0.0], np.where(same_video, np.diff(hip_y) / gap, 0.0)])
    speed = np.concatenate(
        [[0.0], np.where(same_video, np.hypot(np.diff(hip_x), np.diff(hip_y)) / gap, 0.0)]
    )
    d_aspect = np.concatenate([[0.0], np.where(same_video, np.diff(aspect) / gap, 0.0)])

    df["hip_vx"] = np.round(vx, 5)
    df["hip_vy"] = np.round(vy, 5)
    df["hip_speed"] = np.round(speed, 5)
    df["aspect_rate"] = np.round(d_aspect, 5)

    df.to_csv(path, index=False)
    print(f"  recomputed {len(df):,} rows -> {path.name}")


def main() -> None:
    print("Recomputing geometric + motion features from stored landmarks")
    recompute(config.FEATURES_CSV)
    recompute(config.UNSEEN_FEATURES_CSV)


if __name__ == "__main__":
    main()
