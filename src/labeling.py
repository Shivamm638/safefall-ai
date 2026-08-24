"""
SafeFall AI - Stage 2: assign one of the five activity labels to every frame.

Two different sources of truth are combined:

1. **Fall Detected** comes from the Le2i ground-truth annotation files, which
   give the exact frame where each fall starts and ends.  The window is then
   extended for as long as the person remains on the floor, because a fall
   monitor must keep alerting while the resident is still down.  In the
   training set no frame is ever called a fall on posture evidence alone.

2. **Walking / Sitting / Standing / Normal Activity** are not annotated in
   Le2i, so they are derived with a transparent, rule-based (weak supervision)
   labeller that reads the same interpretable geometric features the network
   sees.  Every rule is a statement about body geometry that a clinician could
   check by eye - trunk angle, knee angle, stride width, motion.

The CNN then learns to *generalise* these rules from raw posture, which is why
it can still make a sensible call on frames the rules leave ambiguous.  This
design choice is documented openly as a limitation in the project report.

Run with:  python -m src.labeling
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_PROTECTED_SOURCES = ("ground-truth", "ground-truth-extended")


def assign_labels(df: pd.DataFrame, use_ground_truth_falls: bool = True) -> pd.DataFrame:
    """Add ``activity``, ``label_source`` and ``label_index`` columns.

    Parameters
    ----------
    df
        Frame-level feature table produced by ``build_dataset.py``.
    use_ground_truth_falls
        ``True`` for the annotated subsets, where the fall class is taken
        strictly from the Le2i annotation files.  ``False`` for the
        unseen-scene subsets that ship without annotations - there the fall
        class falls back to the "on the floor" posture rule and the resulting
        labels are treated as a weak reference, never as ground truth.
    """
    out = df.reset_index(drop=True).copy()
    idx = config.CLASS_TO_INDEX

    torso = out["geo_torso_angle"].to_numpy()
    body_axis = out["geo_body_axis_angle"].to_numpy()
    aspect = out["geo_bbox_aspect_ratio"].to_numpy()
    knee = out["geo_mean_knee_angle"].to_numpy()
    hip_angle = out["geo_mean_hip_angle"].to_numpy()
    ankle_sep = out["geo_ankle_separation"].to_numpy()

    # ---------------- posture predicates ----------------------------------- #
    # Every predicate below is decidable from a single frame, because that is
    # exactly what the classifier is given at inference time.
    lying_flat = (aspect > config.LYING_ASPECT_RATIO) & (
        (torso > config.LYING_TORSO_ANGLE)
        | (body_axis > config.LYING_TORSO_ANGLE + 10.0)
    )

    # A resident who falls and ends up slumped against a couch or a wall keeps
    # an upright trunk, so the width/height test above misses them - yet they
    # are just as much on the floor and just as much an emergency. What always
    # changes is *height*: the body occupies far less vertical space than when
    # the same person was standing earlier in the same recording. Comparing
    # against a per-video reference height makes this robust to how far the
    # camera is from the person.
    collapsed = _height_collapsed(out)
    on_the_floor = lying_flat | collapsed

    sitting = (
        (torso < config.SIT_MAX_TORSO_ANGLE)
        & (knee < config.SIT_MAX_KNEE_ANGLE)
        & (hip_angle < config.SIT_MAX_HIP_ANGLE)
        & (aspect < config.SIT_MAX_ASPECT_RATIO)
    )
    walking = (
        (torso < config.WALK_MAX_TORSO_ANGLE)
        & (knee > config.WALK_MIN_KNEE_ANGLE)
        & (ankle_sep >= config.STANCE_WIDTH_BOUNDARY)
        & (aspect < 0.90)
    )
    standing = (
        (torso < config.STAND_MAX_TORSO_ANGLE)
        & (knee > config.STAND_MIN_KNEE_ANGLE)
        & (ankle_sep < config.STANCE_WIDTH_BOUNDARY)
        & (aspect < 0.85)
    )

    # Normal Activity: upright but clearly inclined trunk - bending, reaching,
    # crouching, or mid-transition between sitting and standing.
    bending = (
        (torso >= config.NORMAL_MIN_TORSO_ANGLE)
        & (torso <= config.NORMAL_MAX_TORSO_ANGLE)
        & (aspect < config.LYING_ASPECT_RATIO)
    )

    # ---------------- apply in ascending priority --------------------------- #
    labels = np.full(len(out), idx["Normal Activity"], dtype=np.int16)
    source = np.full(len(out), "rule:normal-residual", dtype=object)

    labels[bending] = idx["Normal Activity"]
    source[bending] = "rule:normal-bending"
    labels[standing] = idx["Standing"]
    source[standing] = "rule:standing"
    labels[walking] = idx["Walking"]
    source[walking] = "rule:walking"
    labels[sitting] = idx["Sitting"]
    source[sitting] = "rule:sitting"

    if use_ground_truth_falls:
        in_fall = out["in_fall_window"].to_numpy().astype(bool)
        after_fall = out["after_fall_window"].to_numpy().astype(bool)

        labels[in_fall] = idx["Fall Detected"]
        source[in_fall] = "ground-truth"

        # Keep alerting while the resident is still down after the annotated
        # fall has finished; if they get back up the rule stops firing.
        still_down = after_fall & on_the_floor
        labels[still_down] = idx["Fall Detected"]
        source[still_down] = "ground-truth-extended"
    else:
        labels[on_the_floor] = idx["Fall Detected"]
        source[on_the_floor] = "posture-rule:fall"

    out["activity"] = [config.CLASS_NAMES[i] for i in labels]
    out["label_source"] = source
    out = _smooth_labels(out)
    out["label_index"] = out["activity"].map(config.CLASS_TO_INDEX).astype(int)
    return out


def _height_collapsed(df: pd.DataFrame) -> np.ndarray:
    """True where the body is far shorter than that person's upright height.

    The reference is taken per video, from the frames before the annotated fall
    where the resident is known to be up and about. Videos with no such frames
    fall back to the 75th percentile of body height across the whole clip,
    which is dominated by the upright portion of any realistic recording.
    """
    height = df["geo_bbox_height"]

    upright = df[(df["in_fall_window"] == 0) & (df["after_fall_window"] == 0)]
    reference = upright.groupby("video_id")["geo_bbox_height"].median()
    fallback = df.groupby("video_id")["geo_bbox_height"].quantile(0.75)
    reference = reference.reindex(fallback.index)
    reference = reference.where(reference.notna() & (reference > 0.05), fallback)

    mapped = df["video_id"].map(reference)
    ratio = height / mapped.replace(0, np.nan)
    return (ratio < config.HEIGHT_COLLAPSE_RATIO).fillna(False).to_numpy()


def _smooth_labels(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Per-video majority filter that removes single-frame label flicker.

    Frames whose label came from the Le2i annotation file are never rewritten.
    """
    out = df.sort_values(["video_id", "frame"]).reset_index(drop=True)
    activity = out["activity"].to_numpy()
    protected = out["label_source"].isin(_PROTECTED_SOURCES).to_numpy()
    video = out["video_id"].to_numpy()

    half = window // 2
    smoothed = activity.copy()
    for i in range(half, len(out) - half):
        if protected[i]:
            continue
        if video[i - half] != video[i] or video[i + half] != video[i]:
            continue
        neighbourhood = activity[i - half: i + half + 1]
        values, counts = np.unique(neighbourhood, return_counts=True)
        if counts.max() > half:
            smoothed[i] = values[int(np.argmax(counts))]

    out["activity"] = smoothed
    return out


def summarise(df: pd.DataFrame, title: str = "Label distribution") -> pd.DataFrame:
    """Per-class frame counts, used by the CLI, the notebook and the report."""
    counts = (
        df["activity"].value_counts().reindex(config.CLASS_NAMES).fillna(0).astype(int)
    )
    table = pd.DataFrame(
        {
            "class": counts.index,
            "frames": counts.to_numpy(),
            "percent": (counts.to_numpy() / max(len(df), 1) * 100).round(2),
        }
    )
    print(f"\n{title}  (total = {len(df):,} frames)")
    print(table.to_string(index=False))
    return table


def main() -> None:
    df = pd.read_csv(config.FEATURES_CSV)
    labelled = assign_labels(df, use_ground_truth_falls=True)
    labelled.to_csv(config.FEATURES_CSV, index=False)
    summarise(labelled, "Annotated subsets (Coffee room 01/02 + Home 01/02)")
    print("\nLabel provenance:")
    print(labelled["label_source"].value_counts().to_string())

    if config.UNSEEN_FEATURES_CSV.exists():
        unseen = pd.read_csv(config.UNSEEN_FEATURES_CSV)
        if len(unseen):
            unseen_labelled = assign_labels(unseen, use_ground_truth_falls=False)
            unseen_labelled.to_csv(config.UNSEEN_FEATURES_CSV, index=False)
            summarise(unseen_labelled, "Unseen scenes (Lecture room + Office)")


if __name__ == "__main__":
    main()
