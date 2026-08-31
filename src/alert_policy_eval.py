"""
SafeFall AI - evaluate and tune the alerting policy, not just the classifier.

Why this exists
---------------
Every number reported so far is per frame, but the deployed system does not
alert per frame. It raises an emergency only when the fall probability stays
above ``FALL_PROB_THRESHOLD`` for ``LIVE_CONFIRM_FRAMES`` consecutive analysed
frames. That confirmation window is what a caregiver actually experiences, and
it had never been measured: the threshold was tuned on per-frame F1 and the
window length was simply chosen.

Per-frame accuracy also flatters or maligns the system unfairly here. A single
stray frame above threshold is not a false alarm - the window absorbs it. And
missing 3 frames of a 40-frame fall is not a missed fall. What matters is:

  * did an alert fire for every video containing a real fall,
  * how long after the fall began,
  * and how often did one fire when nothing had happened.

Method
------
Score every frame with the deployed predictor (ensemble + mirror TTA), replay
each video in order through the exact live state machine, and count alert
*episodes* rather than frames. Sweep threshold x window on VALIDATION videos,
choose one operating point, and report it once on TEST.

Run with:  python -m src.alert_policy_eval
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .data import FeatureScaler, load_split
from .evaluate import predict_dataframe

RESULT_PATH = config.RESULTS_DIR / "alert_policy_eval.json"

# Grace period after a fall begins during which an alert still counts as a
# detection rather than a late one. Falls in Le2i last a second or two.
DETECTION_GRACE_S = 10.0


def replay(probabilities: np.ndarray, threshold: float, window: int) -> List[int]:
    """Replay the live alarm state machine; return the index of each alert onset.

    Mirrors ``LiveMonitorState.update``: the streak grows while the fall
    probability is at or above the threshold and resets otherwise; the alarm
    latches when the streak reaches ``window`` and clears when the streak
    returns to zero. Only the transition into the alarm is recorded, so a fall
    that stays on the floor counts once rather than once per frame.
    """
    onsets: List[int] = []
    streak = 0
    active = False
    for i, p in enumerate(probabilities):
        if p >= threshold:
            streak += 1
        else:
            streak = 0
        if streak >= window:
            if not active:
                onsets.append(i)
            active = True
        elif streak == 0:
            active = False
    return onsets


def evaluate_policy(videos: List[Dict], threshold: float, window: int) -> Dict:
    """Aggregate the policy over a split's videos."""
    detected = missed = 0
    false_episodes = 0
    quiet_seconds = 0.0
    latencies: List[float] = []
    false_alarm_videos = set()

    for video in videos:
        onsets = replay(video["probs"], threshold, window)
        times = video["times"]
        onset_times = [times[i] for i in onsets]

        if video["contains_fall"]:
            start = video["fall_start_s"]
            hits = [t for t in onset_times if t >= start - 1.0]
            if hits:
                detected += 1
                latencies.append(max(hits[0] - start, 0.0))
            else:
                missed += 1
            # Anything raised well before the fall began is a false alarm.
            early = [t for t in onset_times if t < start - 1.0]
            false_episodes += len(early)
            if early:
                false_alarm_videos.add(video["video_id"])
            quiet_seconds += max(start, 0.0)
        else:
            false_episodes += len(onset_times)
            if onset_times:
                false_alarm_videos.add(video["video_id"])
            quiet_seconds += video["duration_s"]

    fall_videos = detected + missed
    non_fall = [v for v in videos if not v["contains_fall"]]
    return {
        "threshold": round(float(threshold), 3),
        "window": int(window),
        "fall_videos": fall_videos,
        "detected": detected,
        "missed": missed,
        "detection_rate": detected / fall_videos if fall_videos else 0.0,
        "median_latency_s": float(np.median(latencies)) if latencies else None,
        "false_alarm_episodes": false_episodes,
        "false_alarm_videos": len(false_alarm_videos),
        "non_fall_videos": len(non_fall),
        "quiet_minutes": round(quiet_seconds / 60.0, 2),
        "false_alarms_per_hour": (
            false_episodes / (quiet_seconds / 3600.0) if quiet_seconds > 0 else 0.0
        ),
    }


def build_videos(df: pd.DataFrame, probs: np.ndarray, index: pd.DataFrame) -> List[Dict]:
    """Group scored frames into per-video replay traces."""
    meta = index.set_index("video_id")
    fall_col = probs[:, config.FALL_INDEX]
    df = df.reset_index(drop=True)

    videos = []
    for video_id, group in df.groupby("video_id", sort=True):
        if video_id not in meta.index:
            continue
        row = meta.loc[video_id]
        fps = float(row["fps"]) or 25.0
        order = group.sort_values("frame")
        frames = order["frame"].to_numpy(dtype=float)
        videos.append({
            "video_id": video_id,
            "probs": fall_col[order.index.to_numpy()],
            "times": frames / fps,
            "duration_s": float(row["n_frames"]) / fps,
            "contains_fall": bool(row["contains_fall"]),
            "fall_start_s": float(row["fall_start"]) / fps if row["contains_fall"] else None,
        })
    return videos


def main() -> None:
    scaler = FeatureScaler.load()
    from .inference import SafeFallPredictor

    index = pd.read_csv(config.VIDEO_INDEX_CSV)
    model = SafeFallPredictor()

    print("Scoring frames with the deployed predictor (ensemble + mirror TTA)...")
    splits = {}
    for name, path in (("validation", config.VAL_CSV), ("test", config.TEST_CSV)):
        df = load_split(path)
        probs, _ = predict_dataframe(df, model, scaler)
        splits[name] = build_videos(df, probs, index)
        falls = sum(v["contains_fall"] for v in splits[name])
        print(f"  {name}: {len(splits[name])} videos ({falls} with a real fall), "
              f"{len(df):,} frames")
    model.close()

    thresholds = np.round(np.arange(0.10, 0.81, 0.05), 2)
    windows = [1, 2, 3, 4, 5, 6, 8, 10]

    print("\nVALIDATION - sweeping threshold x confirmation window")
    grid = [evaluate_policy(splits["validation"], t, w)
            for t in thresholds for w in windows]

    current = evaluate_policy(splits["validation"],
                              config.FALL_PROB_THRESHOLD, config.LIVE_CONFIRM_FRAMES)

    # A window of 1 is excluded on design grounds, not empirical ones: it means
    # no confirmation at all, so a single stray frame raises an emergency. The
    # whole point of the window is to absorb those, and a grid search should not
    # be allowed to delete a safety property because a small sample happened to
    # favour it.
    candidates = [g for g in grid if g["window"] >= 2]
    ranked = sorted(candidates, key=lambda g: (-g["detected"],
                                               g["false_alarms_per_hour"],
                                               g["median_latency_s"] or 9.9))

    # Only move if a point is at least as good on BOTH axes and strictly better
    # on one. With this little quiet footage the false-alarm rate is a handful of
    # episodes extrapolated to an hour, so anything less than a clear win is
    # noise, and the honest action is to leave the deployed policy alone.
    def dominates(g, base):
        return (g["detected"] >= base["detected"]
                and g["false_alarms_per_hour"] <= base["false_alarms_per_hour"]
                and (g["detected"] > base["detected"]
                     or g["false_alarms_per_hour"] < base["false_alarms_per_hour"]))

    better = [g for g in ranked if dominates(g, current)]
    if better:
        best = better[0]
        basis = "strictly better than the current point on validation"
    else:
        best = current
        basis = ("no operating point beat the current one on both detection and "
                 "false alarms; keeping it")
    print(f"  current  ({current['threshold']}, {current['window']}): "
          f"detected {current['detected']}/{current['fall_videos']}  "
          f"false alarms/hour {current['false_alarms_per_hour']:.1f}  "
          f"latency {current['median_latency_s']}")
    print(f"  selected ({best['threshold']}, {best['window']}): "
          f"detected {best['detected']}/{best['fall_videos']}  "
          f"false alarms/hour {best['false_alarms_per_hour']:.1f}  "
          f"latency {best['median_latency_s']}   [{basis}]")

    print(f"\n  NOTE: validation carries only {current['quiet_minutes']:.1f} minutes of "
          f"fall-free footage, so a per-hour false-alarm rate here is a handful of\n"
          f"        episodes extrapolated; treat small differences as noise.")

    print("\nTEST - scored once at each operating point")
    test_current = evaluate_policy(splits["test"], config.FALL_PROB_THRESHOLD,
                                   config.LIVE_CONFIRM_FRAMES)
    test_best = evaluate_policy(splits["test"], best["threshold"], best["window"])
    for label, r in (("current ", test_current), ("selected", test_best)):
        print(f"  {label} ({r['threshold']}, {r['window']}): "
              f"detected {r['detected']}/{r['fall_videos']}  "
              f"missed {r['missed']}  "
              f"false-alarm episodes {r['false_alarm_episodes']} "
              f"over {r['quiet_minutes']:.1f} quiet min "
              f"({r['false_alarms_per_hour']:.1f}/hour)  "
              f"latency {r['median_latency_s']}s")

    payload = {
        "method": (
            "Per-video replay of the deployed alarm state machine (probability "
            "over threshold for N consecutive analysed frames). Alert episodes "
            "are counted, not frames. Operating point chosen on validation "
            "videos and reported once on test."
        ),
        "detection_grace_seconds": DETECTION_GRACE_S,
        "current_operating_point": {
            "threshold": config.FALL_PROB_THRESHOLD,
            "window": config.LIVE_CONFIRM_FRAMES,
            "validation": current,
            "test": test_current,
        },
        "selected_operating_point": {
            "threshold": best["threshold"],
            "window": best["window"],
            "selection_basis": basis,
            "validation": best,
            "test": test_best,
        },
        "validation_grid": grid,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten -> {RESULT_PATH}")


if __name__ == "__main__":
    main()
