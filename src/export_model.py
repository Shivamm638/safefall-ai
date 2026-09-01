"""
SafeFall AI - export the trained Keras model for lightweight deployment.

Writes ``models/fall_detection_cnn_weights.npz`` and then *proves* the NumPy
engine reproduces Keras by comparing both engines on the full test split.
Deployment only goes ahead if the maximum absolute probability difference is
below the float32 tolerance.

Run with:  python -m src.export_model
"""

from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import config
from .data import FeatureScaler, frame_to_arrays, load_split
from .numpy_inference import NumpyPoseCNN, export_weights

TOLERANCE = 1e-4


def export_ensemble(tf, scaler, x_lm, x_geo) -> list:
    """Export every ensemble member and check each one against Keras.

    ``src.inference`` runs the ensemble, not the single model, so these are the
    weights the dashboard actually uses. Exporting only the single model left
    them stale after every retrain, which is why this is done here rather than
    by hand.
    """
    members = sorted(config.MODELS_DIR.glob("ensemble_member_*.keras"))
    if not members:
        print("\nNo ensemble members found - skipping (run `python -m src.experiments`).")
        return []

    print(f"\nExporting {len(members)} ensemble members")
    results = []
    for keras_path in members:
        member = tf.keras.models.load_model(keras_path)
        out_path = keras_path.with_name(keras_path.stem + "_weights.npz")
        export_weights(member, out_path)

        keras_probs = member.predict(
            {"landmarks": x_lm, "geometry": x_geo}, batch_size=512, verbose=0
        )
        numpy_probs = NumpyPoseCNN(out_path).predict(x_lm, x_geo)
        diff = float(np.abs(keras_probs - numpy_probs).max())
        agree = float((keras_probs.argmax(1) == numpy_probs.argmax(1)).mean())
        results.append({"member": keras_path.stem, "max_abs_difference": diff,
                        "class_agreement": agree})
        status = "ok" if diff <= TOLERANCE else "FAILED"
        print(f"  {keras_path.stem:<20} max diff {diff:.3e}  "
              f"agreement {agree:.4%}  [{status}]")
        if diff > TOLERANCE:
            raise SystemExit(f"{keras_path.stem} exceeded the parity tolerance")
    return results


def main() -> None:
    import tensorflow as tf

    model = tf.keras.models.load_model(config.KERAS_MODEL_PATH)
    path = export_weights(model, config.NUMPY_WEIGHTS_PATH)
    size_kb = path.stat().st_size / 1024
    print(f"Exported weights -> {path}  ({size_kb:.0f} KB)")

    # ---- numerical parity check ------------------------------------------- #
    scaler = FeatureScaler.load()
    test_df = load_split(config.TEST_CSV)
    x_lm, x_geo, _ = frame_to_arrays(test_df)
    x_lm, x_geo = scaler.transform(x_lm, x_geo)

    keras_probs = model.predict(
        {"landmarks": x_lm, "geometry": x_geo}, batch_size=512, verbose=0
    )
    numpy_probs = NumpyPoseCNN(path).predict(x_lm, x_geo)

    max_diff = float(np.abs(keras_probs - numpy_probs).max())
    mean_diff = float(np.abs(keras_probs - numpy_probs).mean())
    agreement = float((keras_probs.argmax(1) == numpy_probs.argmax(1)).mean())

    print("\nParity check: Keras engine vs NumPy engine")
    print(f"  samples compared          : {len(x_lm):,}")
    print(f"  max abs probability diff  : {max_diff:.3e}")
    print(f"  mean abs probability diff : {mean_diff:.3e}")
    print(f"  identical predicted class : {agreement:.4%}")

    ensemble_reports = export_ensemble(tf, scaler, x_lm, x_geo)

    report = {
        "ensemble_members": ensemble_reports,
        "samples": int(len(x_lm)),
        "max_abs_difference": max_diff,
        "mean_abs_difference": mean_diff,
        "predicted_class_agreement": agreement,
        "tolerance": TOLERANCE,
        "passed": bool(max_diff < TOLERANCE and agreement == 1.0),
        "weights_file_kb": round(size_kb, 1),
    }
    (config.RESULTS_DIR / "engine_parity_check.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if not report["passed"]:
        raise SystemExit(
            f"\nFAILED: the NumPy engine does not match Keras "
            f"(max diff {max_diff:.3e} >= {TOLERANCE}). Do not deploy."
        )
    print("\nPASSED - the deployed NumPy engine is numerically identical to Keras.")


if __name__ == "__main__":
    main()
