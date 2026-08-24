"""
SafeFall AI - Stage 6: render the visual evidence for the report and the video.

Produces, in ``results/``:

    pose_samples/            clean MediaPipe skeleton output, one per activity
    prediction_samples/      the same frames annotated with the model's verdict
    pose_estimation_grid.png montage of pose output across all five activities
    prediction_grid.png      montage of predictions with confidences
    fall_sequence.png        a real fall unfolding frame by frame

and, in ``data/samples/``, small demo clips and stills the dashboard can be
driven with during the walkthrough video (the raw Le2i .avi files are ~90 MB
each and far too large to ship).

Run with:  python -m src.generate_evidence
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .inference import SafeFallPredictor
from .pose_utils import PoseEstimator

FONT = None  # set lazily, cv2 is imported inside the functions


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _video_lookup() -> Dict[str, str]:
    index = pd.read_csv(config.VIDEO_INDEX_CSV)
    return dict(zip(index["video_id"], index["video_path"]))


def _read_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """Grab a single 1-indexed frame from a video."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 1, 0))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _upscale(frame: np.ndarray, factor: int = 2) -> np.ndarray:
    import cv2

    return cv2.resize(
        frame, (frame.shape[1] * factor, frame.shape[0] * factor),
        interpolation=cv2.INTER_CUBIC,
    )


def _banner(frame: np.ndarray, activity: str, confidence: float,
            truth: Optional[str] = None) -> np.ndarray:
    """Draw the prediction banner used in every evidence screenshot."""
    import cv2

    out = frame.copy()
    height, width = out.shape[:2]
    is_fall = activity == config.FALL_CLASS
    colour = (59, 56, 229) if is_fall else (143, 157, 42)   # BGR

    bar_h = 62
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), colour, -1)
    cv2.addWeighted(overlay, 0.88, out, 0.12, 0, out)

    label = f"{'FALL DETECTED - EMERGENCY' if is_fall else activity.upper()}"
    cv2.putText(out, label, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"confidence {confidence:.1%}", (14, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)

    if truth is not None:
        correct = truth == activity
        tick = "OK" if correct else "X"
        tcol = (120, 190, 90) if correct else (60, 60, 230)
        cv2.putText(out, f"ground truth: {truth} [{tick}]", (width - 330, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, tcol, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (width - 1, height - 1), colour, 3)
    return out


def _montage(images: List[np.ndarray], captions: List[str], columns: int = 3,
             title: str = "") -> np.ndarray:
    """Tile annotated frames into a single labelled contact sheet."""
    import cv2

    if not images:
        return np.zeros((10, 10, 3), np.uint8)

    cell_h = max(img.shape[0] for img in images)
    cell_w = max(img.shape[1] for img in images)
    caption_h = 34
    rows = int(np.ceil(len(images) / columns))
    title_h = 54 if title else 0

    sheet = np.full(
        (title_h + rows * (cell_h + caption_h), columns * cell_w, 3), 248, np.uint8
    )
    if title:
        cv2.putText(sheet, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.92,
                    (40, 35, 27), 2, cv2.LINE_AA)

    for i, (img, caption) in enumerate(zip(images, captions)):
        r, c = divmod(i, columns)
        y = title_h + r * (cell_h + caption_h)
        x = c * cell_w
        sheet[y: y + img.shape[0], x: x + img.shape[1]] = img
        cv2.putText(sheet, caption, (x + 8, y + cell_h + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 55, 50), 1, cv2.LINE_AA)
    return sheet


# --------------------------------------------------------------------------- #
# Evidence generators
# --------------------------------------------------------------------------- #
def _score_test_set(predictor: SafeFallPredictor) -> pd.DataFrame:
    """Predict the whole test split straight from the stored landmarks."""
    from .data import raw_landmarks

    test = pd.read_csv(config.TEST_CSV).reset_index(drop=True)
    probs = predictor.predict_batch(raw_landmarks(test))
    test["predicted"] = [config.CLASS_NAMES[i] for i in probs.argmax(axis=1)]
    test["confidence"] = probs.max(axis=1)
    test["correct"] = test["predicted"] == test["activity"]
    return test


def _render(row, lookup, pose, predictor):
    """Read the source frame and return ``(skeleton, annotated, prediction)``."""
    path = lookup.get(row["video_id"])
    if not path or not Path(path).exists():
        return None
    frame = _read_frame(path, int(row["frame"]))
    if frame is None:
        return None
    frame = _upscale(frame, 2)
    results = pose.process(frame)
    if not results.pose_landmarks:
        return None
    prediction = predictor.predict_image(frame, draw=True, static=True)
    if not prediction.pose_found:
        return None
    return pose.draw(frame, results), prediction


def per_class_samples(predictor: SafeFallPredictor, per_class: int = 2) -> None:
    """Pose screenshots, correct-prediction evidence, and a failure analysis."""
    import cv2

    scored = _score_test_set(predictor)
    lookup = _video_lookup()
    pose = PoseEstimator(static_image_mode=True)

    pose_images, pose_captions = [], []
    pred_images, pred_captions = [], []

    for class_name in config.CLASS_NAMES:
        # Representative frames: correctly classified, confidently, with a
        # clearly visible skeleton. These are the evidence screenshots.
        candidates = scored[(scored["activity"] == class_name) & scored["correct"]]
        if candidates.empty:
            candidates = scored[scored["activity"] == class_name]
        candidates = candidates[candidates["geo_mean_visibility"] > 0.55]
        candidates = candidates.sort_values("confidence", ascending=False).head(40)

        picked = 0
        for _, row in candidates.iterrows():
            if picked >= per_class:
                break
            rendered = _render(row, lookup, pose, predictor)
            if rendered is None:
                continue
            skeleton, prediction = rendered

            slug = class_name.replace(" ", "_").lower()
            cv2.imwrite(str(config.POSE_SAMPLES_DIR / f"pose_{slug}_{picked + 1}.png"), skeleton)
            annotated = _banner(prediction.annotated_image, prediction.activity,
                                prediction.confidence, truth=class_name)
            cv2.imwrite(
                str(config.PREDICTION_SAMPLES_DIR / f"prediction_{slug}_{picked + 1}.png"),
                annotated,
            )

            if picked == 0:
                pose_images.append(skeleton)
                pose_captions.append(f"{class_name} - 33 landmarks detected")
                pred_images.append(annotated)
                pred_captions.append(
                    f"predicted {prediction.activity} ({prediction.confidence:.0%}) "
                    f"| truth {class_name}"
                )
            picked += 1

    cv2.imwrite(
        str(config.RESULTS_DIR / "pose_estimation_grid.png"),
        _montage(pose_images, pose_captions, columns=3,
                 title="MediaPipe Pose output across all five activity classes"),
    )
    cv2.imwrite(
        str(config.RESULTS_DIR / "prediction_grid.png"),
        _montage(pred_images, pred_captions, columns=3,
                 title="SafeFall AI predictions on held-out test frames"),
    )
    print(f"  pose samples      -> {config.POSE_SAMPLES_DIR}")
    print(f"  prediction samples-> {config.PREDICTION_SAMPLES_DIR}")

    _failure_grid(scored, lookup, pose, predictor)
    pose.close()


def _failure_grid(scored: pd.DataFrame, lookup, pose, predictor,
                  limit: int = 6) -> None:
    """The honest half of the evidence: what the model gets wrong, and why."""
    import cv2

    errors = scored[~scored["correct"]].copy()
    if errors.empty:
        return

    # One example of each distinct confusion, most frequent confusions first.
    errors["pair"] = errors["activity"] + " -> " + errors["predicted"]
    order = errors["pair"].value_counts()
    images, captions = [], []

    for pair in order.index[:limit]:
        subset = errors[errors["pair"] == pair]
        subset = subset[subset["geo_mean_visibility"] > 0.5]
        subset = subset.sort_values("confidence", ascending=False)
        for _, row in subset.head(6).iterrows():
            rendered = _render(row, lookup, pose, predictor)
            if rendered is None:
                continue
            _skeleton, prediction = rendered
            images.append(
                _banner(prediction.annotated_image, prediction.activity,
                        prediction.confidence, truth=row["activity"])
            )
            captions.append(
                f"truth {row['activity']} -> predicted {prediction.activity} "
                f"({prediction.confidence:.0%})  [{order[pair]} such frames]"
            )
            break

    if images:
        cv2.imwrite(
            str(config.RESULTS_DIR / "misclassification_examples.png"),
            _montage(images, captions, columns=3,
                     title="Where SafeFall AI gets it wrong - the most common confusions"),
        )
        print(f"  failure analysis  -> "
              f"{config.RESULTS_DIR / 'misclassification_examples.png'}")


def _best_detected_fall(candidates: pd.DataFrame, predictor: SafeFallPredictor,
                        max_checked: int = 8):
    """Pick the test fall video with the strongest mid-fall detection."""
    best_row, best_score = None, -1.0
    for _, row in candidates.head(max_checked).iterrows():
        start, end = int(row["fall_start"]), int(row["fall_end"])
        scores = []
        for number in np.linspace(start, end + 25, 5).round().astype(int):
            frame = _read_frame(row["video_path"], int(number))
            if frame is None:
                continue
            prediction = predictor.predict_image(_upscale(frame, 2), draw=False, static=True)
            if prediction.pose_found:
                scores.append(prediction.probabilities[config.FALL_CLASS])
        score = float(np.mean(scores)) if scores else -1.0
        if score > best_score:
            best_row, best_score = row, score
    return best_row


def fall_sequence(predictor: SafeFallPredictor, n_frames: int = 6) -> None:
    """A real annotated fall unfolding, frame by frame - the headline figure."""
    import cv2

    index = pd.read_csv(config.VIDEO_INDEX_CSV)
    splits = pd.read_csv(config.SPLITS_DIR / "video_split_index.csv")
    test_ids = set(splits[splits["split"] == "test"]["video_id"])

    candidates = index[
        (index["contains_fall"] == 1) & (index["video_id"].isin(test_ids))
    ].sort_values("n_pose_frames", ascending=False)
    if candidates.empty:
        candidates = index[index["contains_fall"] == 1]
    if candidates.empty:
        return

    # Choose a recording the model actually gets right, so the figure shows a
    # representative successful detection rather than an arbitrary clip. The
    # headline metrics elsewhere report the failures honestly.
    row = _best_detected_fall(candidates, predictor)
    if row is None:
        row = candidates.iloc[0]
    start, end = int(row["fall_start"]), int(row["fall_end"])
    span_start = max(start - 25, 1)
    span_end = end + 35
    frame_numbers = np.linspace(span_start, span_end, n_frames).round().astype(int)

    images, captions = [], []
    for number in frame_numbers:
        frame = _read_frame(row["video_path"], int(number))
        if frame is None:
            continue
        frame = _upscale(frame, 2)
        prediction = predictor.predict_image(frame, draw=True, static=True)
        if not prediction.pose_found:
            continue
        images.append(_banner(prediction.annotated_image, prediction.activity,
                              prediction.confidence))
        phase = ("before fall" if number < start
                 else "DURING FALL" if number <= end else "after fall")
        captions.append(f"frame {number} ({number / 25:.1f}s) - {phase}")

    if images:
        cv2.imwrite(
            str(config.RESULTS_DIR / "fall_sequence.png"),
            _montage(images, captions, columns=3,
                     title=f"A real fall detected frame by frame - {row['video_id']} "
                           f"(ground-truth fall: frames {start}-{end})"),
        )
        print(f"  fall sequence     -> {config.RESULTS_DIR / 'fall_sequence.png'}")


def export_demo_media(max_clips: int = 3, clip_seconds: float = 8.0) -> None:
    """Small MP4 clips + stills the dashboard can be demonstrated with."""
    import cv2

    index = pd.read_csv(config.VIDEO_INDEX_CSV)
    splits = pd.read_csv(config.SPLITS_DIR / "video_split_index.csv")
    test_ids = set(splits[splits["split"] == "test"]["video_id"])

    fall_videos = index[(index["contains_fall"] == 1) & (index["video_id"].isin(test_ids))]
    # A genuine activities-of-daily-living clip: an annotated recording whose
    # ground-truth file says no fall occurs. The unannotated Lecture room and
    # Office subsets cannot be used here - several of them do contain falls,
    # which is exactly why they make good unseen-scene test material but bad
    # "this is what normal looks like" demo material.
    adl_videos = index[
        (index["has_annotation"]) & (index["contains_fall"] == 0)
    ].sort_values("n_pose_frames", ascending=False)

    picks: List[tuple] = []
    if not fall_videos.empty:
        for _, row in fall_videos.head(2).iterrows():
            picks.append((row, "fall"))
    if not adl_videos.empty:
        picks.append((adl_videos.iloc[0], "normal"))

    for row, kind in picks[:max_clips]:
        cap = cv2.VideoCapture(row["video_path"])
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        span = int(clip_seconds * fps)
        if kind == "fall" and row["fall_start"] > 0:
            begin = max(int(row["fall_start"]) - int(3 * fps), 0)
        else:
            begin = max((total - span) // 2, 0)
        begin = min(begin, max(total - span, 0))

        out_path = config.SAMPLES_DIR / f"demo_{kind}_{row['video_id']}.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
        for _ in range(span):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        cap.release()

        size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
        print(f"  demo clip         -> {out_path.name} ({size_mb:.1f} MB)")

        # A still from the same recording, for the image-upload demo.
        still_number = (
            int(row["fall_end"]) + 10 if kind == "fall" and row["fall_end"] > 0
            else begin + span // 2
        )
        frame = _read_frame(row["video_path"], still_number)
        if frame is not None:
            cv2.imwrite(
                str(config.SAMPLES_DIR / f"demo_{kind}_{row['video_id']}.jpg"),
                _upscale(frame, 2),
            )


def main() -> None:
    print("Generating visual evidence")
    predictor = SafeFallPredictor()
    print(f"  inference engine  : {predictor.engine_name}")
    per_class_samples(predictor)
    fall_sequence(predictor)
    export_demo_media()
    predictor.close()
    print("Done.")


if __name__ == "__main__":
    main()
