"""
SafeFall AI - runtime for the trained upper-body fall detector.

Loads the weights exported by ``src.upper_body_train`` and runs them in plain
NumPy, for the same reason the main classifier has a NumPy engine: the deployed
app must not carry a deep-learning framework just to make a prediction.

If the weights are missing the module says so rather than guessing, and
``src.inference`` falls back to the original geometric rule, so an incomplete
checkout degrades instead of breaking.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import numpy as np

from . import config
from .pose_utils import (
    UPPER_HEAD_DROP_PER_SEC,
    upper_body_features,
    upper_body_fall_score,
    upper_body_metrics,
)

MODEL_PATH = config.MODELS_DIR / "upper_body_model.json"

_SPEC: Optional[Dict] = None
_LOADED = False


def _load() -> Optional[Dict]:
    global _SPEC, _LOADED
    if not _LOADED:
        _LOADED = True
        if MODEL_PATH.exists():
            try:
                _SPEC = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt file must not break the app
                _SPEC = None
    return _SPEC


def available() -> bool:
    """True when the trained detector can be used."""
    return _load() is not None


def describe() -> str:
    spec = _load()
    if spec is None:
        return "geometric rule"
    return f"trained upper-body {spec['model']['kind']}"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


_ACTIVATIONS = {
    "relu": lambda v: np.maximum(v, 0.0),
    "tanh": np.tanh,
    "logistic": _sigmoid,
    "identity": lambda v: v,
}


def fall_probability(features: np.ndarray) -> float:
    """Probability that this upper-body posture is a person on the floor."""
    spec = _load()
    if spec is None:
        raise RuntimeError("Upper-body model is not available")

    mean = np.asarray(spec["scaler_mean"], dtype=np.float64)
    scale = np.asarray(spec["scaler_scale"], dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    x = (np.asarray(features, dtype=np.float64) - mean) / scale

    model = spec["model"]
    if model["kind"] == "logistic":
        z = float(np.dot(x, np.asarray(model["coef"], dtype=np.float64))
                  + float(model["intercept"]))
        return float(_sigmoid(z))

    hidden = _ACTIVATIONS.get(model.get("activation", "relu"), _ACTIVATIONS["relu"])
    coefs = [np.asarray(c, dtype=np.float64) for c in model["coefs"]]
    biases = [np.asarray(b, dtype=np.float64) for b in model["intercepts"]]
    h = x
    for i, (w, b) in enumerate(zip(coefs, biases)):
        h = h @ w + b
        if i < len(coefs) - 1:
            h = hidden(h)
    z = float(np.ravel(h)[0])
    return float(_sigmoid(z)) if model.get("out_activation") == "logistic" else float(z)


def _to_alarm_scale(probability: float, threshold: float) -> float:
    """Rescale so the model's own threshold lands on ``FALL_PROB_THRESHOLD``.

    The live monitor latches its alarm when the reported fall probability
    crosses ``config.FALL_PROB_THRESHOLD``. The upper-body model's operating
    point is chosen on validation data and is a different number, so reporting
    its raw probability would let the alarm fire at a point the detector never
    agreed to. This is a monotonic piecewise-linear map: it preserves ordering
    and confidence, and makes the two decisions identical by construction.
    """
    cut = config.FALL_PROB_THRESHOLD
    if probability >= threshold:
        span = max(1.0 - threshold, 1e-6)
        return cut + (1.0 - cut) * (probability - threshold) / span
    return cut * probability / max(threshold, 1e-6)


def assess(lm: np.ndarray, head_drop_per_sec: float = 0.0,
           aspect: float = 4.0 / 3.0) -> Tuple[bool, float, Dict]:
    """``(is_fall, confidence, metrics)`` from the upper body alone.

    Matches the signature of the geometric rule it replaces, and falls back to
    it when the trained weights are absent.

    A rapid head drop stays as a second, independent trigger. The classifier
    reads one frame, so it recognises a person who is *already* on the floor;
    the drop term catches the fall while it is still happening, before that
    posture has settled.
    """
    if not available():
        return upper_body_fall_score(lm, head_drop_per_sec, aspect)

    spec = _load()
    features = upper_body_features(lm, aspect)
    probability = fall_probability(features)
    threshold = float(spec.get("threshold", 0.5))

    drop_evidence = float(np.clip(head_drop_per_sec / UPPER_HEAD_DROP_PER_SEC, 0.0, 1.4))
    is_fall = probability >= threshold or drop_evidence >= 1.0

    # Report the probability on a scale where the chosen threshold reads as the
    # halfway point, so "51%" always means "just over the line" whatever the
    # threshold happens to be.
    if probability >= threshold:
        confidence = 0.5 + 0.5 * (probability - threshold) / max(1.0 - threshold, 1e-6)
    else:
        confidence = 0.5 + 0.5 * (threshold - probability) / max(threshold, 1e-6)
    if is_fall and probability < threshold:      # triggered by the drop term
        confidence = max(0.5, min(drop_evidence / 1.4, 0.95))

    metrics = upper_body_metrics(lm, aspect)
    metrics.update({
        "fall_probability": round(probability, 4),
        "alarm_probability": round(_to_alarm_scale(probability, threshold), 4),
        "threshold": threshold,
        "head_drop_per_sec": round(head_drop_per_sec, 3),
        "trunk_angle": round(float(features[2]), 2),
        "hips_visible": bool(features[3] >= 0.5),
        "engine": describe(),
    })
    return bool(is_fall), float(np.clip(confidence, 0.05, 0.99)), metrics
