"""
SafeFall AI - hyperparameter search.

The first model was tuned by hand over a handful of configurations, which is not
the same as searching. This runs a random search over the settings most likely
to matter for this architecture and picks the winner on the VALIDATION split.

Two things are searched that were never varied before:

  * **width** - the network has only 128 k parameters. It may simply be too
    small to separate Normal Activity, which is the class dragging macro F1 down
    (recall 0.40 against 0.86-0.91 for everything else).
  * **augmentation strength** - more augmented copies per frame is a direct
    substitute for the training data this dataset does not have.

The winning configuration is then trained as an ensemble and scored once on
TEST, so the headline number is never selected on the data it reports.

Run with:  python -m src.hparam_search
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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

N_TRIALS = 14
EPOCHS = 40


def sample_config(rng) -> dict:
    return {
        "learning_rate": float(rng.choice([3e-4, 5e-4, 1e-3])),
        "l2": float(rng.choice([3e-5, 1e-4, 3e-4])),
        "dropout_geo": float(rng.choice([0.2, 0.3, 0.4])),
        "dropout_head1": float(rng.choice([0.30, 0.45, 0.55])),
        "dropout_head2": float(rng.choice([0.20, 0.35, 0.45])),
        "width": float(rng.choice([1.0, 1.5, 2.0])),
        "batch_size": int(rng.choice([64, 128])),
        "augment_copies": int(rng.choice([4, 6, 8])),
    }


def train_eval(cfg, data, seed=config.RANDOM_SEED, epochs=EPOCHS):
    """Train one configuration and return (val_accuracy, val_macro_f1, model)."""
    from sklearn.metrics import accuracy_score, f1_score
    from tensorflow.keras import callbacks

    from .model import build_model

    x_lm_tr, x_geo_tr, y_tr = data["train"][cfg["augment_copies"]]
    val_inputs, y_va = data["val"]

    model = build_model(
        learning_rate=cfg["learning_rate"], seed=seed, l2=cfg["l2"],
        dropout_geo=cfg["dropout_geo"], dropout_head1=cfg["dropout_head1"],
        dropout_head2=cfg["dropout_head2"], width=cfg["width"],
    )
    model.fit(
        {"landmarks": x_lm_tr, "geometry": x_geo_tr}, y_tr,
        validation_data=(val_inputs, y_va),
        epochs=epochs, batch_size=cfg["batch_size"],
        class_weight=class_weights(y_tr),
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10,
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                        min_lr=1e-5, verbose=0),
        ],
        verbose=0,
    )
    pred = model.predict(val_inputs, batch_size=512, verbose=0).argmax(1)
    return (float(accuracy_score(y_va, pred)),
            float(f1_score(y_va, pred, average="macro", zero_division=0)),
            model)


def main() -> None:
    rng = np.random.RandomState(config.RANDOM_SEED)

    train_df = load_split(config.TRAIN_CSV)
    val_df = load_split(config.VAL_CSV)
    scaler = FeatureScaler.load()

    balanced = balance_by_class(train_df)
    raw_tr = raw_landmarks(balanced)
    y_raw = balanced["label_index"].to_numpy(dtype=np.int64)

    # Augmented sets are expensive to build, so build one per distinct copy count
    # and reuse it across every trial that asks for it.
    print("Preparing augmented training sets...")
    train_sets = {}
    for copies in (4, 6, 8):
        x_lm, x_geo, y = build_augmented_set(raw_tr, y_raw, copies=copies)
        x_lm, x_geo = scaler.transform(x_lm, x_geo)
        train_sets[copies] = (x_lm, x_geo, y)
        print(f"  copies={copies}: {len(y):,} samples")

    x_lm_va, x_geo_va, y_va = frame_to_arrays(val_df)
    x_lm_va, x_geo_va = scaler.transform(x_lm_va, x_geo_va)
    data = {"train": train_sets,
            "val": ({"landmarks": x_lm_va, "geometry": x_geo_va}, y_va)}

    baseline = {
        "learning_rate": config.LEARNING_RATE, "l2": config.L2_REGULARISATION,
        "dropout_geo": 0.30, "dropout_head1": 0.45, "dropout_head2": 0.35,
        "width": 1.0, "batch_size": config.BATCH_SIZE,
        "augment_copies": config.AUGMENT_COPIES,
    }

    trials = [("baseline", baseline)]
    for i in range(N_TRIALS):
        trials.append((f"trial-{i + 1}", sample_config(rng)))

    print(f"\nRunning {len(trials)} configurations (validation-selected)\n")
    results = []
    t_start = time.time()
    for name, cfg in trials:
        t0 = time.time()
        acc, macro_f1, _ = train_eval(cfg, data)
        results.append({"name": name, "config": cfg, "val_accuracy": acc,
                        "val_macro_f1": macro_f1})
        print(f"  {name:<10} acc {acc:.4f}  macroF1 {macro_f1:.4f}  "
              f"[w{cfg['width']} lr{cfg['learning_rate']:.0e} "
              f"do{cfg['dropout_head1']:.2f} aug{cfg['augment_copies']}] "
              f"({time.time() - t0:.0f}s)")

    results.sort(key=lambda r: r["val_accuracy"], reverse=True)
    best = results[0]
    base = next(r for r in results if r["name"] == "baseline")

    print(f"\nSearch finished in {(time.time() - t_start) / 60:.1f} min")
    print(f"  baseline validation accuracy : {base['val_accuracy']:.4f}")
    print(f"  best     validation accuracy : {best['val_accuracy']:.4f}  ({best['name']})")
    print(f"  best config: {json.dumps(best['config'])}")

    (config.RESULTS_DIR / "hparam_search.json").write_text(
        json.dumps({"trials": results, "best": best, "baseline": base}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten -> {config.RESULTS_DIR / 'hparam_search.json'}")


if __name__ == "__main__":
    main()
