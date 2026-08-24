"""
SafeFall AI - dataset loading and feature scaling.

Turns the frame-level CSV tables into the two tensors the network expects:

    X_landmarks : (N, 33, 4)  hip-centred, torso-scaled skeleton
    X_geometry  : (N, 25)     interpretable posture descriptors

The scaler is stored as plain JSON (not a pickle) so the deployed dashboard can
reproduce the exact training-time normalisation without needing scikit-learn -
and so the numbers stay readable and auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import config

LANDMARK_COLUMNS = [
    f"lm{j}_{axis}"
    for j in range(config.NUM_LANDMARKS)
    for axis in ("x", "y", "z", "v")
]
GEOMETRY_COLUMNS = [f"geo_{name}" for name in config.GEOMETRIC_FEATURE_NAMES]


@dataclass
class FeatureScaler:
    """Per-feature standardisation for both network inputs."""

    lm_mean: np.ndarray      # (33, 4)
    lm_std: np.ndarray       # (33, 4)
    geo_mean: np.ndarray     # (25,)
    geo_std: np.ndarray      # (25,)

    @classmethod
    def fit(cls, x_lm: np.ndarray, x_geo: np.ndarray) -> "FeatureScaler":
        return cls(
            lm_mean=x_lm.mean(axis=0),
            lm_std=np.maximum(x_lm.std(axis=0), 1e-6),
            geo_mean=x_geo.mean(axis=0),
            geo_std=np.maximum(x_geo.std(axis=0), 1e-6),
        )

    def transform(
        self, x_lm: np.ndarray, x_geo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            ((x_lm - self.lm_mean) / self.lm_std).astype(np.float32),
            ((x_geo - self.geo_mean) / self.geo_std).astype(np.float32),
        )

    def save(self, path: Path = config.SCALER_PATH) -> None:
        payload = {
            "landmark_shape": list(self.lm_mean.shape),
            "lm_mean": self.lm_mean.tolist(),
            "lm_std": self.lm_std.tolist(),
            "geo_mean": self.geo_mean.tolist(),
            "geo_std": self.geo_std.tolist(),
            "geometric_feature_names": config.GEOMETRIC_FEATURE_NAMES,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = config.SCALER_PATH) -> "FeatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            lm_mean=np.asarray(payload["lm_mean"], dtype=np.float32),
            lm_std=np.asarray(payload["lm_std"], dtype=np.float32),
            geo_mean=np.asarray(payload["geo_mean"], dtype=np.float32),
            geo_std=np.asarray(payload["geo_std"], dtype=np.float32),
        )


def raw_landmarks(df: pd.DataFrame) -> np.ndarray:
    """The unmodified MediaPipe landmarks as ``(N, 33, 4)`` in image coordinates.

    Augmentation works on these, because geometric features such as trunk angle
    have to be *recomputed* from a transformed skeleton rather than transformed
    themselves.
    """
    raw = df[LANDMARK_COLUMNS].to_numpy(dtype=np.float32)
    return raw.reshape(-1, config.NUM_LANDMARKS, config.LANDMARK_CHANNELS)


def frame_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract ``(X_landmarks, X_geometry, y)`` from a feature table.

    The landmark columns hold *raw* MediaPipe coordinates, so the same
    hip-centring / torso-scaling used at inference time is applied here.
    """
    from .pose_utils import normalize_landmarks

    raw = df[LANDMARK_COLUMNS].to_numpy(dtype=np.float32)
    raw = raw.reshape(-1, config.NUM_LANDMARKS, config.LANDMARK_CHANNELS)
    x_lm = np.stack([normalize_landmarks(sample) for sample in raw]).astype(np.float32)

    x_geo = df[GEOMETRY_COLUMNS].to_numpy(dtype=np.float32)
    x_geo = np.nan_to_num(x_geo, nan=0.0, posinf=50.0, neginf=-50.0)

    y = None
    if "label_index" in df.columns:
        y = df["label_index"].to_numpy(dtype=np.int64)
    return x_lm, x_geo, y


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label_index" not in df.columns and "activity" in df.columns:
        df["label_index"] = df["activity"].map(config.CLASS_TO_INDEX).astype(int)
    return df


def balance_by_class(
    df: pd.DataFrame, max_per_class: int = config.MAX_SAMPLES_PER_CLASS, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Cap over-represented classes in the *training* split only.

    Validation and test splits are always left untouched so the reported
    metrics reflect the real class distribution a monitor would see.
    """
    parts = []
    rng = np.random.RandomState(seed)
    for label, group in df.groupby("activity", sort=False):
        if len(group) > max_per_class:
            keep = rng.choice(group.index.to_numpy(), size=max_per_class, replace=False)
            parts.append(group.loc[np.sort(keep)])
        else:
            parts.append(group)
    return pd.concat(parts).sort_index()


def class_weights(y: np.ndarray) -> dict:
    """Inverse-frequency weights so rare-but-critical falls are not ignored."""
    counts = np.bincount(y, minlength=config.NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (config.NUM_CLASSES * counts)
    return {i: float(w) for i, w in enumerate(weights)}
