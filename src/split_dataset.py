"""
SafeFall AI - Stage 3: the 70 / 15 / 15 train-validation-test split.

The split is made **at video level, not frame level**.  Neighbouring frames of
the same recording are almost identical, so a random frame-wise split would put
near-duplicates on both sides of the fence and report an accuracy that could
never be reproduced on a new resident.  Splitting whole videos keeps the test
set genuinely unseen.

Videos are stratified by scene subset and by whether they contain a fall, so
all three partitions carry a comparable share of emergencies.

Run with:  python -m src.split_dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def _video_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per video with the attributes used for stratification."""
    table = (
        df.groupby("video_id")
        .agg(
            subset=("subset", "first"),
            frames=("frame", "size"),
            fall_frames=("activity", lambda s: int((s == config.FALL_CLASS).sum())),
        )
        .reset_index()
    )
    table["contains_fall"] = (table["fall_frames"] > 0).astype(int)
    table["stratum"] = table["subset"] + "_" + table["contains_fall"].astype(str)
    return table


def _safe_strata(strata: pd.Series, min_count: int):
    """Merge strata that are too small to split, or give up on stratifying.

    ``train_test_split`` refuses to stratify when any group has fewer members
    than the number of partitions, so rare scene/fall combinations are first
    merged into a shared bucket; if even that bucket stays too small the split
    falls back to a plain random one.
    """
    counts = strata.value_counts()
    rare = set(counts[counts < min_count].index)
    merged = strata.where(~strata.isin(rare), other="other")
    if (merged.value_counts() < min_count).any():
        return None
    return merged


def make_splits(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Return the video table with a ``split`` column added."""
    videos = _video_table(df)

    train_ids, holdout_ids = train_test_split(
        videos["video_id"],
        train_size=config.TRAIN_RATIO,
        random_state=seed,
        stratify=_safe_strata(videos["stratum"], 3),
    )
    holdout_mask = videos["video_id"].isin(holdout_ids)

    val_ids, test_ids = train_test_split(
        videos.loc[holdout_mask, "video_id"],
        train_size=config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=seed,
        stratify=_safe_strata(videos.loc[holdout_mask, "stratum"], 2),
    )

    split = pd.Series("train", index=videos.index)
    split[videos["video_id"].isin(val_ids)] = "val"
    split[videos["video_id"].isin(test_ids)] = "test"
    videos["split"] = split.to_numpy()
    return videos


def main() -> None:
    df = pd.read_csv(config.FEATURES_CSV)
    if "activity" not in df.columns:
        raise SystemExit("Run `python -m src.labeling` before splitting.")

    videos = make_splits(df)
    df = df.merge(videos[["video_id", "split"]], on="video_id", how="left")

    for name, path in (
        ("train", config.TRAIN_CSV),
        ("val", config.VAL_CSV),
        ("test", config.TEST_CSV),
    ):
        part = df[df["split"] == name]
        part.to_csv(path, index=False)

    videos.to_csv(config.SPLITS_DIR / "video_split_index.csv", index=False)

    # ---- report ----------------------------------------------------------- #
    n_videos = len(videos)
    print("Video-level split (70 / 15 / 15)")
    print("-" * 64)
    for name in ("train", "val", "test"):
        vids = videos[videos["split"] == name]
        frames = int(df[df["split"] == name].shape[0])
        print(
            f"  {name:<6} videos={len(vids):>4} ({len(vids) / n_videos:5.1%})   "
            f"frames={frames:>7,} ({frames / len(df):5.1%})   "
            f"fall videos={int(vids['contains_fall'].sum()):>3}"
        )

    print("\nFrames per class in each split")
    print("-" * 64)
    pivot = (
        df.pivot_table(
            index="activity", columns="split", values="frame", aggfunc="size", fill_value=0
        )
        .reindex(config.CLASS_NAMES)
        .fillna(0)
        .astype(int)
    )
    pivot = pivot[[c for c in ("train", "val", "test") if c in pivot.columns]]
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.to_string())

    leakage = set(videos[videos.split == "train"].video_id) & set(
        videos[videos.split != "train"].video_id
    )
    print(f"\nVideo overlap between train and held-out sets: {len(leakage)} (must be 0)")


if __name__ == "__main__":
    main()
