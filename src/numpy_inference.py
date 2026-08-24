"""
SafeFall AI - NumPy execution engine for the trained CNN.

Why this file exists
--------------------
The network is trained with TensorFlow/Keras, but TensorFlow is a ~600 MB
dependency and Streamlit Community Cloud gives an app only 1 GB of RAM - which
TensorFlow, MediaPipe and OpenCV together will not reliably fit inside.

Rather than shrink the model or give up on deployment, the trained weights are
exported and the *same* architecture is re-implemented here in pure NumPy.  It
is the identical network doing the identical arithmetic: ``verify_parity.py``
checks the two engines agree to within float32 round-off on the whole test set.

The deployed dashboard therefore runs the real trained model on ~40 MB of
dependencies instead of ~700 MB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

BN_EPSILON = 1e-3          # Keras BatchNormalization default


# --------------------------------------------------------------------------- #
# Layer primitives
# --------------------------------------------------------------------------- #
def conv1d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """1-D convolution, stride 1, ``padding='same'`` (odd kernel sizes).

    x      : (N, L, C_in)
    kernel : (K, C_in, C_out)
    """
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    # (N, L, C_in, K) -> (N, L, K, C_in)
    windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=1)
    windows = np.ascontiguousarray(windows.transpose(0, 1, 3, 2))
    return np.einsum("nlkc,kco->nlo", windows, kernel, optimize=True) + bias


def batch_norm(x: np.ndarray, gamma, beta, mean, var, eps: float = BN_EPSILON) -> np.ndarray:
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def max_pool1d(x: np.ndarray, pool: int = 2) -> np.ndarray:
    """Keras ``MaxPooling1D(pool_size=pool)`` with the default 'valid' padding."""
    n, length, channels = x.shape
    usable = (length // pool) * pool
    return x[:, :usable, :].reshape(n, usable // pool, pool, channels).max(axis=2)


def dense(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ kernel + bias


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class NumpyPoseCNN:
    """Forward pass of ``SafeFall_PoseCNN`` using nothing but NumPy."""

    def __init__(self, weights_path: Path):
        data = np.load(str(weights_path))
        self.w: Dict[str, np.ndarray] = {k: data[k].astype(np.float32) for k in data.files}

    def _bn(self, x: np.ndarray, name: str) -> np.ndarray:
        return batch_norm(
            x,
            self.w[f"{name}_gamma"],
            self.w[f"{name}_beta"],
            self.w[f"{name}_mean"],
            self.w[f"{name}_var"],
        )

    def predict(self, x_lm: np.ndarray, x_geo: np.ndarray) -> np.ndarray:
        """``x_lm`` (N, 33, 4) and ``x_geo`` (N, 25), already scaled -> (N, 5)."""
        x_lm = np.asarray(x_lm, dtype=np.float32)
        x_geo = np.asarray(x_geo, dtype=np.float32)
        if x_lm.ndim == 2:
            x_lm = x_lm[None, ...]
        if x_geo.ndim == 1:
            x_geo = x_geo[None, ...]

        # -- Branch A: convolutions along the kinematic chain ---------------- #
        h = relu(conv1d_same(x_lm, self.w["conv1_kernel"], self.w["conv1_bias"]))
        h = self._bn(h, "bn1")
        h = relu(conv1d_same(h, self.w["conv2_kernel"], self.w["conv2_bias"]))
        h = self._bn(h, "bn2")
        h = max_pool1d(h, 2)
        h = relu(conv1d_same(h, self.w["conv3_kernel"], self.w["conv3_bias"]))
        h = self._bn(h, "bn3")
        pose = np.concatenate([h.mean(axis=1), h.max(axis=1)], axis=-1)

        # -- Branch B: clinical descriptors ---------------------------------- #
        g = relu(dense(x_geo, self.w["geo_dense_kernel"], self.w["geo_dense_bias"]))
        g = self._bn(g, "geo_bn")

        # -- Fusion head (dropout is identity at inference) ------------------ #
        z = np.concatenate([pose, g], axis=-1)
        z = relu(dense(z, self.w["head1_kernel"], self.w["head1_bias"]))
        z = relu(dense(z, self.w["head2_kernel"], self.w["head2_bias"]))
        logits = dense(z, self.w["activity_kernel"], self.w["activity_bias"])
        return softmax(logits).astype(np.float32)

    # keep the call signature interchangeable with a Keras model
    def __call__(self, inputs, *_args, **_kwargs) -> np.ndarray:
        return self.predict(inputs["landmarks"], inputs["geometry"])


class NumpyEnsemble:
    """Average several NumpyPoseCNN members.

    Independently seeded copies of the same architecture make different
    mistakes, so averaging their probabilities beats any single member. Each
    member is only 469 KB and a forward pass is sub-millisecond, so the whole
    ensemble costs far less than the pose estimator that feeds it.
    """

    def __init__(self, weight_paths):
        self.members = [NumpyPoseCNN(p) for p in weight_paths]
        if not self.members:
            raise ValueError("an ensemble needs at least one member")

    def predict(self, x_lm: np.ndarray, x_geo: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(x_lm, x_geo) for m in self.members], axis=0)

    def __call__(self, inputs, *_args, **_kwargs) -> np.ndarray:
        return self.predict(inputs["landmarks"], inputs["geometry"])


# --------------------------------------------------------------------------- #
# Weight export (run once, after training)
# --------------------------------------------------------------------------- #
LAYER_EXPORT_PLAN = {
    "conv1": ("kernel", "bias"),
    "bn1": ("gamma", "beta", "mean", "var"),
    "conv2": ("kernel", "bias"),
    "bn2": ("gamma", "beta", "mean", "var"),
    "conv3": ("kernel", "bias"),
    "bn3": ("gamma", "beta", "mean", "var"),
    "geo_dense": ("kernel", "bias"),
    "geo_bn": ("gamma", "beta", "mean", "var"),
    "head1": ("kernel", "bias"),
    "head2": ("kernel", "bias"),
    "activity": ("kernel", "bias"),
}


def export_weights(keras_model, out_path: Path) -> Path:
    """Flatten a trained Keras model into a ``.npz`` the engine above can read."""
    payload: Dict[str, np.ndarray] = {}
    for layer_name, suffixes in LAYER_EXPORT_PLAN.items():
        layer = keras_model.get_layer(layer_name)
        weights = layer.get_weights()
        if len(weights) != len(suffixes):
            raise ValueError(
                f"Layer '{layer_name}' returned {len(weights)} weight arrays, "
                f"expected {len(suffixes)}"
            )
        for suffix, array in zip(suffixes, weights):
            payload[f"{layer_name}_{suffix}"] = np.asarray(array, dtype=np.float32)

    out_path = Path(out_path)
    np.savez_compressed(out_path, **payload)
    return out_path
