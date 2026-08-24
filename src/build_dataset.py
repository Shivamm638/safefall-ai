"""
SafeFall AI - Stage 1: turn raw Le2i videos into a pose-feature table.

For every video we sample one frame in ``config.FRAME_STRIDE``, run MediaPipe
Pose, and store:

  * the 33 raw landmarks (x, y, z, visibility)
  * 25 engineered geometric posture features
  * frame-to-frame motion features (used for labelling, not by the model)
  * the ground-truth fall window taken from the Le2i annotation files

Labelling is deliberately NOT done here.  Extraction is expensive and runs
once; labelling is cheap and lives in ``apply_labels.py`` so thresholds can be
re-tuned without re-processing 17 GB of video.

Run with:  python -m src.build_dataset
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .pose_utils import PoseEstimator, geometric_feature_vector, pose_is_usable

# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
ANNOTATION_DIR_NAMES = ("Annotation_files", "Annotations_files", "Annotation files")

# Subsets that ship with frame-level fall annotations -> supervised training set.
ANNOTATED_SUBSETS = {"Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"}


def _subset_name(video_path: Path) -> str:
    """Map any video path to the Le2i subset it belongs to."""
    parts = [p.replace(" ", "_") for p in video_path.parts]
    for known in ("Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02",
                  "Lecture_room", "Office"):
        if known in parts:
            return known
    return video_path.parent.name.replace(" ", "_")


def find_annotation(video_path: Path) -> Optional[Path]:
    """Locate ``video (i).txt`` for a given ``video (i).avi``."""
    stem = video_path.stem
    for parent in (video_path.parent, video_path.parent.parent):
        for name in ANNOTATION_DIR_NAMES:
            candidate = parent / name / f"{stem}.txt"
            if candidate.exists():
                return candidate
        sibling = parent / f"{stem}.txt"
        if sibling.exists():
            return sibling
    return None


def parse_annotation(path: Optional[Path]) -> Tuple[int, int]:
    """Return ``(fall_start_frame, fall_end_frame)``; ``(0, 0)`` means no fall.

    The Le2i ground-truth file begins with two integers: the frame where the
    fall starts and the frame where it ends (1-indexed).
    """
    if path is None or not path.exists():
        return 0, 0
    try:
        with open(path, "r", errors="ignore") as fh:
            head = [fh.readline().strip() for _ in range(2)]
        start, end = int(float(head[0])), int(float(head[1]))
        if start <= 0 or end <= start:
            return 0, 0
        return start, end
    except Exception:
        return 0, 0


def discover_videos(root: Path) -> List[Dict]:
    """Walk the raw dataset and build an index of every ``.avi`` with metadata."""
    videos: List[Dict] = []
    for path in sorted(root.rglob("*.avi")):
        subset = _subset_name(path)
        ann = find_annotation(path)
        fall_start, fall_end = parse_annotation(ann)
        videos.append(
            {
                "video_path": str(path),
                "video_id": f"{subset}__{path.stem}".replace(" ", "_"),
                "subset": subset,
                "has_annotation": ann is not None,
                "fall_start": fall_start,
                "fall_end": fall_end,
                "contains_fall": int(fall_start > 0),
                "split_group": "annotated" if subset in ANNOTATED_SUBSETS else "unseen_scene",
            }
        )
    return videos


# --------------------------------------------------------------------------- #
# Per-video worker
# --------------------------------------------------------------------------- #
_WORKER_POSE: Optional[PoseEstimator] = None


def _init_worker() -> None:
    """Give every process its own MediaPipe graph (they are not shareable)."""
    global _WORKER_POSE
    os.environ.setdefault("GLOG_minloglevel", "3")
    _WORKER_POSE = PoseEstimator(static_image_mode=False)


def process_video(meta: Dict) -> Tuple[Dict, List[Dict]]:
    """Extract pose features for one video. Returns ``(meta_out, rows)``."""
    import cv2

    global _WORKER_POSE
    if _WORKER_POSE is None:
        _init_worker()
    pose = _WORKER_POSE

    stride = config.FRAME_STRIDE
    rows: List[Dict] = []
    cap = cv2.VideoCapture(meta["video_path"])
    if not cap.isOpened():
        meta = {**meta, "n_frames": 0, "n_pose_frames": 0, "fps": 0.0, "status": "unreadable"}
        return meta, rows

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    kept: List[Dict] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1                       # 1-indexed to match the annotations
        if (frame_idx - 1) % stride != 0:
            continue

        lm = pose.extract(frame)
        if not pose_is_usable(lm):
            continue

        geo = geometric_feature_vector(lm)
        kept.append({"frame": frame_idx, "lm": lm, "geo": geo})

    cap.release()

    # ---- motion features across the sampled sequence ---------------------- #
    dt = stride / max(fps, 1.0)
    hip_idx_y = config.GEOMETRIC_FEATURE_NAMES.index("hip_y")
    aspect_idx = config.GEOMETRIC_FEATURE_NAMES.index("bbox_aspect_ratio")

    for i, item in enumerate(kept):
        lm = item["lm"]
        hip_x = float((lm[23, 0] + lm[24, 0]) / 2.0)
        hip_y = float(item["geo"][hip_idx_y])

        if i == 0:
            vx = vy = speed = d_aspect = 0.0
        else:
            prev = kept[i - 1]
            prev_hip_x = float((prev["lm"][23, 0] + prev["lm"][24, 0]) / 2.0)
            prev_hip_y = float(prev["geo"][hip_idx_y])
            gap = max(item["frame"] - prev["frame"], 1) / max(fps, 1.0)
            vx = (hip_x - prev_hip_x) / gap
            vy = (hip_y - prev_hip_y) / gap          # positive == moving downward
            speed = float(np.hypot(hip_x - prev_hip_x, hip_y - prev_hip_y) / gap)
            d_aspect = (item["geo"][aspect_idx] - prev["geo"][aspect_idx]) / gap

        row: Dict[str, float] = {
            "video_id": meta["video_id"],
            "subset": meta["subset"],
            "frame": item["frame"],
            "in_fall_window": int(
                meta["fall_start"] > 0
                and meta["fall_start"] <= item["frame"] <= meta["fall_end"]
            ),
            "after_fall_window": int(
                meta["fall_end"] > 0 and item["frame"] > meta["fall_end"]
            ),
            "hip_vx": round(vx, 5),
            "hip_vy": round(vy, 5),
            "hip_speed": round(speed, 5),
            "aspect_rate": round(float(d_aspect), 5),
        }
        for j, name in enumerate(config.GEOMETRIC_FEATURE_NAMES):
            row[f"geo_{name}"] = round(float(item["geo"][j]), 5)
        flat = lm.reshape(-1)
        for j in range(config.NUM_LANDMARKS):
            row[f"lm{j}_x"] = round(float(flat[j * 4 + 0]), 5)
            row[f"lm{j}_y"] = round(float(flat[j * 4 + 1]), 5)
            row[f"lm{j}_z"] = round(float(flat[j * 4 + 2]), 5)
            row[f"lm{j}_v"] = round(float(flat[j * 4 + 3]), 5)
        rows.append(row)

    meta_out = {
        **meta,
        "n_frames": frame_idx,
        "n_pose_frames": len(rows),
        "fps": round(float(fps), 2),
        "status": "ok",
    }
    return meta_out, rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SafeFall pose-feature dataset")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=0, help="debug: only N videos")
    args = parser.parse_args()

    root = config.RAW_DATASET_DIR
    if not root.exists():
        sys.exit(f"Raw dataset not found at {root}")

    videos = discover_videos(root)
    if args.limit:
        videos = videos[: args.limit]

    annotated = [v for v in videos if v["split_group"] == "annotated"]
    unseen = [v for v in videos if v["split_group"] == "unseen_scene"]
    print(f"Discovered {len(videos)} videos "
          f"({len(annotated)} annotated, {len(unseen)} unseen-scene)")
    print(f"Videos containing a ground-truth fall: "
          f"{sum(v['contains_fall'] for v in videos)}")
    print(f"Using {args.workers} worker processes, frame stride = {config.FRAME_STRIDE}")

    t0 = time.time()
    metas: List[Dict] = []
    rows_annotated: List[Dict] = []
    rows_unseen: List[Dict] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for i, (meta_out, rows) in enumerate(
            pool.imap_unordered(process_video, videos, chunksize=1), start=1
        ):
            metas.append(meta_out)
            if meta_out["split_group"] == "annotated":
                rows_annotated.extend(rows)
            else:
                rows_unseen.extend(rows)
            if i % 10 == 0 or i == len(videos):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                print(f"  [{i}/{len(videos)}] {elapsed:6.1f}s "
                      f"({rate:4.2f} videos/s) rows={len(rows_annotated) + len(rows_unseen)}",
                      flush=True)

    pd.DataFrame(metas).to_csv(config.VIDEO_INDEX_CSV, index=False)
    pd.DataFrame(rows_annotated).to_csv(config.FEATURES_CSV, index=False)
    if rows_unseen:
        pd.DataFrame(rows_unseen).to_csv(config.UNSEEN_FEATURES_CSV, index=False)

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"  annotated rows : {len(rows_annotated):,}  -> {config.FEATURES_CSV}")
    print(f"  unseen rows    : {len(rows_unseen):,}  -> {config.UNSEEN_FEATURES_CSV}")
    print(f"  video index    : {config.VIDEO_INDEX_CSV}")


if __name__ == "__main__":
    main()
