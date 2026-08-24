"""
SafeFall AI - accuracy experiments.

Two cheap, standard ways to squeeze more out of the trained architecture,
measured properly before either is adopted:

  * **Mirror test-time augmentation.** Every geometric feature the model uses is
    an angle or a ratio, so a mirrored skeleton is the same posture. Averaging
    the prediction for a pose and its mirror image cancels part of the model's
    left/right bias at the cost of one extra forward pass.

  * **Seed ensembling.** The same architecture trained from different random
    initialisations makes different mistakes; averaging their probabilities
    usually beats any single member.

Both are selected on the VALIDATION split. The test split is scored once at the
end, so the headline number stays honest.

Run with:  python -m src.experiments
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from . import config
from .augment import build_augmented_set, mirror
from .data import (
    FeatureScaler,
    balance_by_class,
    class_weights,
    frame_to_arrays,
    load_split,
    raw_landmarks,
)
from .pose_utils import geometric_feature_vector, normalize_landmarks


def _inputs_from_raw(raw: np.ndarray, scaler: FeatureScaler):
    lm = np.stack([normalize_landmarks(s) for s in raw]).astype(np.float32)
    geo = np.stack([geometric_feature_vector(s) for s in raw]).astype(np.float32)
    return scaler.transform(lm, geo)


def predict(model, raw: np.ndarray, scaler: FeatureScaler, tta: bool) -> np.ndarray:
    """Probabilities for a batch of raw skeletons, optionally mirror-averaged."""
    x_lm, x_geo = _inputs_from_raw(raw, scaler)
    probs = model.predict({"landmarks": x_lm, "geometry": x_geo},
                          batch_size=512, verbose=0)
    if not tta:
        return probs
    mirrored = np.stack([mirror(s) for s in raw])
    m_lm, m_geo = _inputs_from_raw(mirrored, scaler)
    probs_m = model.predict({"landmarks": m_lm, "geometry": m_geo},
                            batch_size=512, verbose=0)
    return (probs + probs_m) / 2.0


def train_one(seed: int, x_lm, x_geo, y, val_data, epochs: int):
    """Train a single member with the production recipe but a different seed."""
    import tensorflow as tf
    from tensorflow.keras import callbacks

    tf.keras.utils.set_random_seed(seed)
    from .model import build_model

    model = build_model(seed=seed)
    model.fit(
        {"landmarks": x_lm, "geometry": x_geo}, y,
        validation_data=val_data,
        epochs=epochs, batch_size=config.BATCH_SIZE,
        class_weight=class_weights(y),
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                    patience=config.EARLY_STOPPING_PATIENCE,
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=5, min_lr=1e-5, verbose=0),
        ],
        verbose=0,
    )
    return model


def main() -> None:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    seeds = [config.RANDOM_SEED, 7, 2024]
    epochs = config.EPOCHS

    train_df = load_split(config.TRAIN_CSV)
    val_df = load_split(config.VAL_CSV)
    test_df = load_split(config.TEST_CSV)

    balanced = balance_by_class(train_df)
    raw_tr = raw_landmarks(balanced)
    y_raw = balanced["label_index"].to_numpy(dtype=np.int64)
    x_lm_tr, x_geo_tr, y_tr = build_augmented_set(raw_tr, y_raw,
                                                  copies=config.AUGMENT_COPIES)

    scaler = FeatureScaler.load()
    # build_augmented_set returns normalised-but-unscaled tensors. Training on
    # those while predicting on scaled ones is a train/inference mismatch that
    # collapses the model onto a single class - the geometric branch alone spans
    # angles in degrees and ratios near 1.
    x_lm_tr, x_geo_tr = scaler.transform(x_lm_tr, x_geo_tr)
    x_lm_va, x_geo_va, y_va = frame_to_arrays(val_df)
    x_lm_va, x_geo_va = scaler.transform(x_lm_va, x_geo_va)
    val_data = ({"landmarks": x_lm_va, "geometry": x_geo_va}, y_va)

    raw_va = raw_landmarks(val_df)
    raw_te = raw_landmarks(test_df)
    y_te = test_df["label_index"].to_numpy(dtype=np.int64)

    print(f"Training {len(seeds)} members on {len(y_tr):,} augmented samples\n")
    models = []
    for seed in seeds:
        model = train_one(seed, x_lm_tr, x_geo_tr, y_tr, val_data, epochs)
        acc = accuracy_score(y_va, predict(model, raw_va, scaler, False).argmax(1))
        print(f"  seed {seed:<5} validation accuracy {acc:.4f}")
        models.append(model)

    def score(probs, y_true):
        pred = probs.argmax(1)
        return {
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
            "fall_recall": recall_score(y_true, pred, labels=[config.FALL_INDEX],
                                        average="micro", zero_division=0),
        }

    variants = {
        "single": lambda raw, tta=False: predict(models[0], raw, scaler, False),
        "single + mirror TTA": lambda raw: predict(models[0], raw, scaler, True),
        "ensemble of 3": lambda raw: np.mean(
            [predict(m, raw, scaler, False) for m in models], axis=0),
        "ensemble + mirror TTA": lambda raw: np.mean(
            [predict(m, raw, scaler, True) for m in models], axis=0),
    }

    print("\nVALIDATION - used to choose")
    print(f"  {'variant':<24}{'accuracy':>10}{'macro F1':>10}{'fall recall':>13}")
    val_scores = {}
    for name, fn in variants.items():
        s = score(fn(raw_va), y_va)
        val_scores[name] = s
        print(f"  {name:<24}{s['accuracy']:>10.4f}{s['macro_f1']:>10.4f}"
              f"{s['fall_recall']:>13.4f}")

    best = max(val_scores, key=lambda k: val_scores[k]["accuracy"])
    print(f"\n  -> best on validation: {best}")

    print("\nTEST - scored once, with the variant chosen above")
    test_scores = {}
    for name, fn in variants.items():
        test_scores[name] = score(fn(raw_te), y_te)
    print(f"  {'variant':<24}{'accuracy':>10}{'macro F1':>10}{'fall recall':>13}")
    for name, s in test_scores.items():
        marker = "  <- selected" if name == best else ""
        print(f"  {name:<24}{s['accuracy']:>10.4f}{s['macro_f1']:>10.4f}"
              f"{s['fall_recall']:>13.4f}{marker}")

    report = {
        "seeds": seeds,
        "selected_on_validation": best,
        "validation": val_scores,
        "test": test_scores,
    }
    (config.RESULTS_DIR / "experiments.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )

    # Persist the ensemble members so the app can use them if they won.
    if best.startswith("ensemble"):
        for i, model in enumerate(models):
            model.save(config.MODELS_DIR / f"ensemble_member_{i}.keras")
        print(f"\nSaved {len(models)} ensemble members to {config.MODELS_DIR}")

    print(f"\nWritten -> {config.RESULTS_DIR / 'experiments.json'}")


if __name__ == "__main__":
    main()
