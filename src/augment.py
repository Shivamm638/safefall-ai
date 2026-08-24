"""
SafeFall AI - pose-space data augmentation.

Augmentation is applied to the *raw* MediaPipe landmarks and both network
inputs are then recomputed from the augmented skeleton, so the convolutional
branch and the geometric branch always describe the same body.

Each transform mimics a real failure mode of a ceiling- or wall-mounted camera
in a care home:

    mirror     - the resident walks the other way down the corridor
    rotation   - the camera bracket is not perfectly level
    scaling    - the resident is nearer to or further from the lens
    translation- the resident is in a different part of the room
    jitter     - normal landmark noise from the pose estimator
    occlusion  - furniture hides a limb, so some landmarks lose confidence

This is what turns a model that memorises 91 videos into one that generalises.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import config
from .pose_utils import geometric_feature_vector, normalize_landmarks

# MediaPipe left/right landmark pairs, needed to mirror a skeleton correctly:
# flipping x alone would leave a left elbow labelled as a left elbow on what is
# now the right-hand side of the body.
_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

_MIRROR_INDEX = np.arange(config.NUM_LANDMARKS)
for _a, _b in _MIRROR_PAIRS:
    _MIRROR_INDEX[_a], _MIRROR_INDEX[_b] = _b, _a


def mirror(lm: np.ndarray) -> np.ndarray:
    out = lm[_MIRROR_INDEX].copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def rotate(lm: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.radians(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    centre = lm[:, :2].mean(axis=0)
    out = lm.copy()
    shifted = lm[:, :2] - centre
    out[:, 0] = shifted[:, 0] * cos - shifted[:, 1] * sin + centre[0]
    out[:, 1] = shifted[:, 0] * sin + shifted[:, 1] * cos + centre[1]
    return out


def scale(lm: np.ndarray, factor: float) -> np.ndarray:
    centre = lm[:, :2].mean(axis=0)
    out = lm.copy()
    out[:, :2] = (lm[:, :2] - centre) * factor + centre
    out[:, 2] = lm[:, 2] * factor
    return out


def translate(lm: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = lm.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out


def jitter(lm: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = lm.copy()
    out[:, :2] += rng.normal(0.0, sigma, size=(config.NUM_LANDMARKS, 2))
    return out


def occlude(lm: np.ndarray, n_landmarks: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a limb hidden behind furniture: low confidence plus drift."""
    out = lm.copy()
    victims = rng.choice(config.NUM_LANDMARKS, size=n_landmarks, replace=False)
    out[victims, 3] *= rng.uniform(0.05, 0.35)
    out[victims, :2] += rng.normal(0.0, 0.02, size=(n_landmarks, 2))
    return out


def augment_one(lm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random combination of transforms to one raw skeleton.

    Only *label-preserving* transforms are used at full strength. Mirroring,
    scaling and translation leave every quantity the labels depend on (trunk
    angle, knee angle, bounding-box aspect ratio, stance width) mathematically
    unchanged, because those are all angles or ratios.

    Rotation is the exception: tilting a skeleton changes its trunk angle,
    which is the very signal that separates "upright" from "on the floor". A
    large rotation would therefore relabel the sample without relabelling its
    target - injecting label noise rather than robustness. It is kept at a few
    degrees only, to cover a slightly off-level camera bracket.
    """
    out = lm.copy()
    if rng.random() < 0.50:
        out = mirror(out)
    if rng.random() < 0.40:
        out = rotate(out, rng.uniform(-3.0, 3.0))
    if rng.random() < 0.75:
        out = scale(out, rng.uniform(0.80, 1.25))
    if rng.random() < 0.70:
        out = translate(out, rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06))
    if rng.random() < 0.80:
        out = jitter(out, rng.uniform(0.003, 0.012), rng)
    if rng.random() < 0.30:
        out = occlude(out, int(rng.integers(1, 4)), rng)
    return out.astype(np.float32)


def build_augmented_set(
    raw_landmarks: np.ndarray,
    labels: np.ndarray,
    copies: int = 3,
    seed: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a training set with ``copies`` augmented versions of every frame.

    Returns ``(X_landmarks, X_geometry, y)`` with the originals first.
    """
    rng = np.random.default_rng(seed)

    all_raw = [raw_landmarks]
    all_y = [labels]
    for _ in range(copies):
        augmented = np.stack([augment_one(sample, rng) for sample in raw_landmarks])
        all_raw.append(augmented)
        all_y.append(labels)

    stacked = np.concatenate(all_raw, axis=0)
    y = np.concatenate(all_y, axis=0)

    x_lm = np.stack([normalize_landmarks(sample) for sample in stacked]).astype(np.float32)
    x_geo = np.stack([geometric_feature_vector(sample) for sample in stacked]).astype(np.float32)
    return x_lm, x_geo, y
