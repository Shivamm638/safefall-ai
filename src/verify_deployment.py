"""
SafeFall AI - pre-deployment self-check.

Streamlit Community Cloud installs only ``requirements.txt``. Locally, however,
TensorFlow and scikit-learn are also installed for training, so it is very easy
to write app code that quietly depends on a package the deployed environment
will not have - and only find out when the public link is already in the
submission.

This script simulates the deployed environment by making every
training-only package unimportable, then drives the complete pipeline: model
load, pose estimation, image prediction, video analysis and alert logic.

It also checks that every file the app needs is actually present.

Run with:  python -m src.verify_deployment
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

# Packages that appear in requirements-dev.txt but NOT in requirements.txt.
# matplotlib and jax are deliberately absent from this list: mediapipe declares
# both as hard dependencies, so the deployed environment really will have them.
TRAINING_ONLY = {
    "tensorflow",
    "tensorflow_cpu",
    "keras",
    "sklearn",
    "seaborn",
    "joblib",
}


class _DeploymentSimulator:
    """Import hook that hides training-only packages.

    ``ModuleNotFoundError`` is raised rather than a plain ``ImportError``
    because that is what Python raises for a genuinely absent package, and
    libraries guard their optional imports against that specific type -
    mediapipe's own optional TensorFlow import is one such case.
    """

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in TRAINING_ONLY:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


def check_required_files() -> bool:
    from . import config

    required = [
        (config.NUMPY_WEIGHTS_PATH, "exported network weights"),
        (config.SCALER_PATH, "feature scaler"),
        (config.METADATA_PATH, "model card"),
        (config.RESULTS_DIR / "metrics_summary.json", "evaluation metrics"),
        (config.RESULTS_DIR / "confusion_matrix.png", "confusion matrix"),
        (config.RESULTS_DIR / "accuracy_graph.png", "accuracy graph"),
        (config.RESULTS_DIR / "loss_graph.png", "loss graph"),
        (Path(config.PROJECT_ROOT) / "requirements.txt", "runtime requirements"),
        (Path(config.PROJECT_ROOT) / "packages.txt", "apt packages"),
    ]
    ok = True
    print("Required files")
    for path, label in required:
        exists = Path(path).exists()
        size = f"{Path(path).stat().st_size / 1024:8.1f} KB" if exists else "  MISSING"
        print(f"  {'OK ' if exists else 'FAIL'}  {label:<28} {size}")
        ok &= exists
    return ok


def main() -> None:
    sys.meta_path.insert(0, _DeploymentSimulator())

    if not check_required_files():
        raise SystemExit("\nFAILED: files the deployed app needs are missing.")

    import cv2  # noqa: F401

    from . import config
    from .inference import SafeFallPredictor

    print("\nSimulating the deployed environment "
          f"(hidden: {', '.join(sorted(TRAINING_ONLY))})")

    predictor = SafeFallPredictor()
    print(f"  OK    model loaded via {predictor.engine_name}")
    print(f"  OK    alert threshold {config.FALL_PROB_THRESHOLD}")

    images = sorted(glob.glob(str(config.SAMPLES_DIR / "*.jpg")))
    videos = sorted(glob.glob(str(config.SAMPLES_DIR / "*.mp4")))
    if not images or not videos:
        raise SystemExit("FAILED: no demo media - run `python -m src.generate_evidence`.")

    for path in images:
        frame = cv2.imread(path)
        result = predictor.predict_image(frame, draw=True, static=True)
        if result.annotated_image is None:
            raise SystemExit(f"FAILED: no pose overlay produced for {path}")
        status = "EMERGENCY" if result.is_fall else "safe"
        print(f"  OK    image  {Path(path).name[:44]:<44} -> "
              f"{result.activity:<16}{result.confidence:6.1%} [{status}]")

    for path in videos:
        analysis = predictor.analyse_video(path)
        summary = analysis.summary
        print(f"  OK    video  {Path(path).name[:44]:<44} -> "
              f"{summary['fall_events']} fall event(s), "
              f"dominant {summary['dominant_activity']}")

    predictor.close()

    leaked = sorted(m for m in sys.modules if m.split(".")[0] in TRAINING_ONLY)
    if leaked:
        raise SystemExit(
            f"\nFAILED: the app imported training-only packages: {leaked}\n"
            "Either add them to requirements.txt or remove the dependency."
        )

    print("\nPASSED - the dashboard runs on requirements.txt alone.")
    print("Safe to deploy. See docs/DEPLOYMENT_GUIDE.md for the next steps.")


if __name__ == "__main__":
    main()
