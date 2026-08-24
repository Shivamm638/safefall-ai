"""
SafeFall AI - Stage 4: train the activity classifier.

Run with:  python -m src.train
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import config
from .augment import build_augmented_set
from .data import (
    FeatureScaler,
    balance_by_class,
    class_weights,
    frame_to_arrays,
    load_split,
    raw_landmarks,
)
from .model import build_model, summary_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SafeFall pose CNN")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--augment-copies", type=int, default=config.AUGMENT_COPIES)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras import callbacks

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    # ------------------------------------------------------------------ data
    train_df = load_split(config.TRAIN_CSV)
    val_df = load_split(config.VAL_CSV)

    balanced = balance_by_class(train_df)
    print(f"Training frames : {len(train_df):,} -> {len(balanced):,} after capping "
          f"each class at {config.MAX_SAMPLES_PER_CLASS:,}")
    print(f"Validation frames: {len(val_df):,}")

    # Augmentation happens on the raw skeletons, and both network inputs are
    # rebuilt from the transformed body - never on the validation split.
    raw_tr = raw_landmarks(balanced)
    y_raw = balanced["label_index"].to_numpy(dtype=np.int64)
    x_lm_tr, x_geo_tr, y_tr = build_augmented_set(raw_tr, y_raw, copies=args.augment_copies)
    print(f"After augmentation: {len(y_raw):,} -> {len(y_tr):,} training samples "
          f"({args.augment_copies} augmented copies per frame)")

    x_lm_va, x_geo_va, y_va = frame_to_arrays(val_df)

    scaler = FeatureScaler.fit(x_lm_tr, x_geo_tr)
    scaler.save()
    x_lm_tr, x_geo_tr = scaler.transform(x_lm_tr, x_geo_tr)
    x_lm_va, x_geo_va = scaler.transform(x_lm_va, x_geo_va)

    print("\nClass balance used for training")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name:<16} {int((y_tr == i).sum()):>7,}")

    weights = None if args.no_class_weights else class_weights(y_tr)
    if weights:
        print("\nClass weights (inverse frequency, keeps rare falls influential)")
        for i, name in enumerate(config.CLASS_NAMES):
            print(f"  {name:<16} {weights[i]:.3f}")

    # ----------------------------------------------------------------- model
    model = build_model(learning_rate=args.lr)
    print("\n" + summary_text(model))
    (config.RESULTS_DIR / "model_architecture.txt").write_text(
        summary_text(model), encoding="utf-8"
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(config.KERAS_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        {"landmarks": x_lm_tr, "geometry": x_geo_tr},
        y_tr,
        validation_data=({"landmarks": x_lm_va, "geometry": x_geo_va}, y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=weights,
        callbacks=cbs,
        verbose=2,
    )
    train_seconds = time.time() - t0

    # --------------------------------------------------------------- persist
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(config.HISTORY_CSV, index=False)

    model.save(config.KERAS_MODEL_PATH)

    best_epoch = int(hist["val_accuracy"].idxmax()) + 1
    metadata = {
        "model_name": model.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "framework": f"TensorFlow {tf.__version__} / Keras {tf.keras.__version__}",
        "python": platform.python_version(),
        "class_names": config.CLASS_NAMES,
        "geometric_feature_names": config.GEOMETRIC_FEATURE_NAMES,
        "landmark_shape": [config.NUM_LANDMARKS, config.LANDMARK_CHANNELS],
        "total_parameters": int(model.count_params()),
        "epochs_run": int(len(hist)),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(hist["val_accuracy"].max()),
        "final_train_accuracy": float(hist["accuracy"].iloc[-1]),
        "train_frames": int(len(balanced)),
        "val_frames": int(len(val_df)),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "class_weighted": weights is not None,
        "training_seconds": round(train_seconds, 1),
        "split_ratio": [config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO],
        "split_granularity": "video-level (no frame leakage between splits)",
        "frame_stride": config.FRAME_STRIDE,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nTrained {len(hist)} epochs in {train_seconds:.1f}s")
    print(f"Best validation accuracy {hist['val_accuracy'].max():.4f} (epoch {best_epoch})")
    print(f"Model    -> {config.KERAS_MODEL_PATH}")
    print(f"Scaler   -> {config.SCALER_PATH}")
    print(f"History  -> {config.HISTORY_CSV}")


if __name__ == "__main__":
    main()
