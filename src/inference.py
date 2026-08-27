"""
SafeFall AI - the end-to-end fall-detection pipeline used by the dashboard.

    image / video frame
        -> MediaPipe Pose            (33 body landmarks)
        -> feature engineering       (normalised skeleton + 25 posture features)
        -> CNN classifier            (activity + confidence)
        -> temporal confirmation     (video only: smoothing + rapid-descent check)
        -> emergency decision        (alert vs safe)

The same class is used for a single uploaded photo, a webcam snapshot and a
full video, so the behaviour a caregiver sees is always the behaviour that was
evaluated.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import config
from .data import FeatureScaler
from .pose_utils import (
    PoseEstimator,
    build_model_inputs,
    check_framing,
    compute_geometric_features,
    pose_is_usable,
    upper_body_fall_score,
)
from .augment import mirror


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class FramePrediction:
    """The model's verdict on one frame."""

    activity: str
    confidence: float
    probabilities: Dict[str, float]
    is_fall: bool
    pose_found: bool
    geometry: Dict[str, float] = field(default_factory=dict)
    annotated_image: Optional[np.ndarray] = None
    message: str = ""
    framing_ok: bool = True
    framing_score: float = 1.0
    framing_advice: str = ""
    engine: str = "full-body CNN"

    @property
    def status(self) -> str:
        if not self.pose_found:
            return "NO PERSON"
        if not self.framing_ok:
            return "OUT OF FRAME"
        return "EMERGENCY" if self.is_fall else "SAFE"


@dataclass
class VideoAnalysis:
    """Everything the dashboard needs after processing an uploaded video."""

    timeline: List[Dict]
    summary: Dict
    fall_events: List[Dict]
    key_frames: List[Dict]
    duration_seconds: float
    processed_frames: int
    fps: float


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class SafeFallPredictor:
    """Loads the trained model once and serves predictions."""

    def __init__(self, prefer_tensorflow: bool = False, static_image_mode: bool = False):
        # One predictor is shared across every Streamlit script run via
        # @st.cache_resource, and each run executes on its own thread. The live
        # camera loop can therefore still be mid-frame when the rerun that
        # replaces it starts. MediaPipe graphs are native and not thread-safe,
        # so overlapping access crashes the whole process with no traceback -
        # this lock serialises every path that touches one.
        self._lock = threading.RLock()
        self.scaler = FeatureScaler.load(config.SCALER_PATH)
        self.engine_name, self.model = self._load_engine(prefer_tensorflow)
        self._pose_video: Optional[PoseEstimator] = None
        self._pose_image: Optional[PoseEstimator] = None
        self._static_default = static_image_mode
        self._video_complexity = config.POSE_MODEL_COMPLEXITY

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_engine(prefer_tensorflow: bool):
        """Prefer the light NumPy engine; fall back to Keras if asked or needed."""
        if prefer_tensorflow:
            try:
                import tensorflow as tf

                return "TensorFlow/Keras", tf.keras.models.load_model(config.KERAS_MODEL_PATH)
            except Exception:
                pass

        members = sorted(config.MODELS_DIR.glob("ensemble_member_*_weights.npz"))
        if len(members) > 1:
            from .numpy_inference import NumpyEnsemble

            return (f"NumPy ensemble of {len(members)} (exported Keras weights)",
                    NumpyEnsemble(members))

        if config.NUMPY_WEIGHTS_PATH.exists():
            from .numpy_inference import NumpyPoseCNN

            return "NumPy (exported Keras weights)", NumpyPoseCNN(config.NUMPY_WEIGHTS_PATH)

        import tensorflow as tf

        return "TensorFlow/Keras", tf.keras.models.load_model(config.KERAS_MODEL_PATH)

    def _pose(self, static: bool) -> PoseEstimator:
        if static:
            if self._pose_image is None:
                self._pose_image = PoseEstimator(static_image_mode=True)
            return self._pose_image
        if self._pose_video is None:
            self._pose_video = PoseEstimator(
                static_image_mode=False, model_complexity=self._video_complexity
            )
        return self._pose_video

    def reset_video_tracker(self, complexity: Optional[int] = None) -> None:
        """Prepare for a new video or live session.

        This used to close the MediaPipe graph and build a fresh one every time
        a session started. That was the main source of instability: tearing down
        a native graph while any other thread might still be inside it crashes
        the whole process with no Python traceback, and Streamlit starts every
        script run on a new thread, so "any other thread" is the normal case.

        The rebuild bought nothing. MediaPipe's tracker keeps a region-of-interest
        prior from the previous frame, and that prior self-corrects within a
        frame or two of new footage - which the warm-up frames already absorb.

        So the graph is now reused for the lifetime of the process, and is only
        rebuilt when the pose model itself has to change (the quality selector),
        which is rare and never happens mid-stream.
        """
        with self._lock:
            if complexity is not None and complexity != self._video_complexity:
                # Genuinely a different network - the graph must be replaced.
                if self._pose_video is not None:
                    self._pose_video.close()
                    self._pose_video = None
                self._video_complexity = complexity

    # ------------------------------------------------------------- prediction
    def predict_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Landmark array (33, 4) -> probability vector (5,).

        The pose and its mirror image are scored together as a **batch of two**
        rather than in two separate calls. With a 3-member ensemble the naive
        version made six batch-of-one forward passes and cost 16 ms per frame -
        more than half of MediaPipe's own budget. Batching drops that to three
        passes and recomputes the mirrored features once instead of per member.
        """
        poses = [landmarks]
        if config.USE_MIRROR_TTA:
            poses.append(mirror(landmarks))

        tensors, vectors = [], []
        for pose in poses:
            t, v = build_model_inputs(pose)
            tensors.append(t)
            vectors.append(v)
        x_lm, x_geo = self.scaler.transform(np.stack(tensors), np.stack(vectors))

        if hasattr(self.model, "trainable_variables"):
            probs = self.model({"landmarks": x_lm, "geometry": x_geo}, training=False)
        else:
            probs = self.model.predict(x_lm, x_geo)
        return np.asarray(probs, dtype=np.float32).mean(axis=0)

    def predict_batch(self, landmark_batch: np.ndarray) -> np.ndarray:
        """Landmark batch (N, 33, 4) -> probabilities (N, 5)."""
        tensors, vectors = [], []
        for lm in landmark_batch:
            t, v = build_model_inputs(lm)
            tensors.append(t)
            vectors.append(v)
        x_lm, x_geo = self.scaler.transform(np.stack(tensors), np.stack(vectors))
        if hasattr(self.model, "trainable_variables"):
            probs = self.model.predict(
                {"landmarks": x_lm, "geometry": x_geo}, batch_size=256, verbose=0
            )
        else:
            probs = self.model.predict(x_lm, x_geo)
        return np.asarray(probs, dtype=np.float32)

    def predict_image(
        self,
        frame_bgr: np.ndarray,
        draw: bool = True,
        static: Optional[bool] = None,
        enforce_framing: bool = False,
        head_drop_per_sec: float = 0.0,
    ) -> FramePrediction:
        """Full single-frame pipeline, including the pose overlay."""
        with self._lock:
            pose = self._pose(self._static_default if static is None else static)
            results = pose.process(frame_bgr)
            landmarks = pose.landmarks_to_array(results)
            annotated = pose.draw(frame_bgr, results) if draw else None

        fallback_image = annotated if annotated is not None else (
            frame_bgr.copy() if draw else None
        )

        if landmarks is None:
            return FramePrediction(
                activity="No person detected",
                confidence=0.0,
                probabilities={name: 0.0 for name in config.CLASS_NAMES},
                is_fall=False,
                pose_found=False,
                annotated_image=fallback_image,
                message=(
                    "No human body was found. Step into the camera's view, with "
                    "reasonable lighting."
                ),
                framing_ok=False,
                framing_score=0.0,
                framing_advice="No person detected.",
            )

        # A skeleton exists. Judge the framing first: the CNN needs hips, knees and
        # ankles, so when those are out of shot it must not be asked - MediaPipe
        # invents their positions and any answer would be computed from
        # coordinates that do not exist.
        #
        # Rather than refuse, fall back to an upper-body detector that uses only
        # what is genuinely visible: head axis, shoulder tilt and head drop. It
        # answers the question that actually matters - on the floor or not -
        # while being clear that a different engine produced it.
        framing = check_framing(landmarks)
        if not framing.ok:
            height, width = frame_bgr.shape[:2]
            is_fall, confidence, metrics = upper_body_fall_score(
                landmarks, head_drop_per_sec, aspect=width / max(height, 1)
            )
            activity = config.FALL_CLASS if is_fall else "Normal Activity"
            probabilities = {name: 0.0 for name in config.CLASS_NAMES}
            probabilities[config.FALL_CLASS] = confidence if is_fall else 1.0 - confidence
            probabilities["Normal Activity"] = 1.0 - probabilities[config.FALL_CLASS]
            return FramePrediction(
                activity=activity,
                confidence=confidence,
                probabilities=probabilities,
                is_fall=is_fall,
                pose_found=True,
                geometry=compute_geometric_features(landmarks),
                annotated_image=fallback_image,
                message=(
                    f"Upper-body mode: only head and shoulders are in shot, so the "
                    f"full-body classifier cannot be used. Head axis "
                    f"{metrics['head_axis_angle']:.0f}\u00b0 from vertical, shoulder "
                    f"tilt {metrics['shoulder_tilt']:.0f}\u00b0"
                    + (" - consistent with a person on the floor."
                       if is_fall else " - consistent with an upright person.")
                ),
                framing_ok=False,
                framing_score=framing.score,
                framing_advice=framing.advice,
                engine="upper-body geometry",
            )

        if not pose_is_usable(landmarks):
            return FramePrediction(
                activity="No person detected",
                confidence=0.0,
                probabilities={name: 0.0 for name in config.CLASS_NAMES},
                is_fall=False,
                pose_found=False,
                annotated_image=fallback_image,
                message=(
                    "The detected body is too indistinct to classify. Improve the "
                    "lighting, or move so you are not heavily occluded."
                ),
                framing_ok=False,
                framing_score=framing.score,
                framing_advice=framing.advice,
            )

        probs = self.predict_landmarks(landmarks)
        index = int(np.argmax(probs))
        activity = config.CLASS_NAMES[index]
        confidence = float(probs[index])
        geometry = compute_geometric_features(landmarks)

        is_fall = (
            activity == config.FALL_CLASS
            and float(probs[config.FALL_INDEX]) >= config.FALL_PROB_THRESHOLD
        )

        return FramePrediction(
            activity=activity,
            confidence=confidence,
            probabilities={
                name: float(probs[i]) for i, name in enumerate(config.CLASS_NAMES)
            },
            is_fall=is_fall,
            pose_found=True,
            geometry=geometry,
            annotated_image=annotated,
            message=self._explain(activity, geometry, is_fall),
            framing_ok=framing.ok,
            framing_score=framing.score,
            framing_advice=framing.advice,
        )

    @staticmethod
    def _explain(activity: str, geometry: Dict[str, float], is_fall: bool) -> str:
        """One plain-English sentence a caregiver can act on."""
        torso = geometry.get("torso_angle", 0.0)
        aspect = geometry.get("bbox_aspect_ratio", 0.0)
        knee = geometry.get("mean_knee_angle", 0.0)

        if is_fall:
            return (
                f"Body is close to horizontal (trunk tilted {torso:.0f} deg from upright, "
                f"body width/height ratio {aspect:.2f}). This is the posture signature of "
                "a person on the floor - emergency response required."
            )
        if activity == "Sitting":
            return (
                f"Trunk is upright ({torso:.0f} deg) with the knees bent "
                f"({knee:.0f} deg) - a seated posture. No action needed."
            )
        if activity == "Walking":
            return (
                f"Upright trunk with legs apart in a stride pattern "
                f"(trunk {torso:.0f} deg, knees {knee:.0f} deg) - normal mobility."
            )
        if activity == "Standing":
            return (
                f"Upright and stable (trunk {torso:.0f} deg, knees {knee:.0f} deg) "
                "with feet close together. No action needed."
            )
        return (
            f"Upright posture that does not match a specific activity pattern "
            f"(trunk {torso:.0f} deg, knees {knee:.0f} deg) - treated as safe, "
            "routine movement."
        )

    # ------------------------------------------------------------------ video
    def analyse_video(
        self,
        video_path: str | Path,
        max_frames: int = config.MAX_VIDEO_FRAMES,
        stride: int = config.VIDEO_FRAME_STRIDE,
        progress_callback: Optional[Callable[[float], None]] = None,
        keep_key_frames: int = 6,
    ) -> VideoAnalysis:
        """Run the pipeline across a video and apply temporal fall confirmation."""
        import cv2

        self.reset_video_tracker()
        pose = self._pose(static=False)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("The uploaded video could not be opened.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        budget = min(max_frames, (total_frames // max(stride, 1)) or max_frames)

        timeline: List[Dict] = []
        frames_cache: List[np.ndarray] = []
        landmark_batch: List[np.ndarray] = []
        meta: List[Dict] = []

        frame_index = 0
        processed = 0
        prev_hip_y: Optional[float] = None
        prev_hip_x: Optional[float] = None
        prev_time: Optional[float] = None
        t_start = time.time()

        while processed < budget:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            timestamp = frame_index / max(fps, 1.0)
            results = pose.process(frame)
            landmarks = pose.landmarks_to_array(results)

            if pose_is_usable(landmarks):
                geometry = compute_geometric_features(landmarks)
                hip_y = geometry["hip_y"]
                hip_x = float((landmarks[23, 0] + landmarks[24, 0]) / 2.0)
                if prev_hip_y is not None and prev_time is not None:
                    gap = max(timestamp - prev_time, 1e-3)
                    descent = (hip_y - prev_hip_y) / gap
                    speed = float(
                        np.hypot(hip_x - prev_hip_x, hip_y - prev_hip_y) / gap
                    )
                else:
                    descent = speed = 0.0
                prev_hip_y, prev_hip_x, prev_time = hip_y, hip_x, timestamp

                landmark_batch.append(landmarks)
                meta.append(
                    {
                        "frame": frame_index,
                        "time": round(timestamp, 2),
                        "descent_velocity": round(float(descent), 4),
                        "hip_speed": round(float(speed), 4),
                        "torso_angle": round(geometry["torso_angle"], 1),
                        "aspect_ratio": round(geometry["bbox_aspect_ratio"], 3),
                        "pose_found": True,
                    }
                )
                frames_cache.append(pose.draw(frame, results))
            else:
                meta.append(
                    {
                        "frame": frame_index,
                        "time": round(timestamp, 2),
                        "descent_velocity": 0.0,
                        "hip_speed": 0.0,
                        "torso_angle": 0.0,
                        "aspect_ratio": 0.0,
                        "pose_found": False,
                    }
                )
                frames_cache.append(frame.copy())

            processed += 1
            frame_index += 1
            if progress_callback and budget:
                progress_callback(min(processed / budget, 1.0))

        cap.release()

        # ---- one batched forward pass over every detected pose ------------- #
        probs_by_position: Dict[int, np.ndarray] = {}
        if landmark_batch:
            batch_probs = self.predict_batch(np.stack(landmark_batch))
            detected_positions = [i for i, m in enumerate(meta) if m["pose_found"]]
            for pos, row in zip(detected_positions, batch_probs):
                probs_by_position[pos] = row

        for pos, item in enumerate(meta):
            probs = probs_by_position.get(pos)
            if probs is None:
                timeline.append(
                    {
                        **item,
                        "activity": "No person detected",
                        "confidence": 0.0,
                        "fall_probability": 0.0,
                    }
                )
                continue
            index = int(np.argmax(probs))
            timeline.append(
                {
                    **item,
                    "activity": config.CLASS_NAMES[index],
                    "confidence": round(float(probs[index]), 4),
                    "fall_probability": round(float(probs[config.FALL_INDEX]), 4),
                    **{
                        f"p_{config.CLASS_NAMES[i]}": round(float(probs[i]), 4)
                        for i in range(config.NUM_CLASSES)
                    },
                }
            )

        self._refine_with_motion(timeline)

        fall_events, smoothed = self._confirm_falls(timeline)
        for row, value in zip(timeline, smoothed):
            row["fall_probability_smoothed"] = round(float(value), 4)

        summary = self._summarise(timeline, fall_events, fps, time.time() - t_start)
        key_frames = self._pick_key_frames(timeline, frames_cache, fall_events, keep_key_frames)

        return VideoAnalysis(
            timeline=timeline,
            summary=summary,
            fall_events=fall_events,
            key_frames=key_frames,
            duration_seconds=timeline[-1]["time"] if timeline else 0.0,
            processed_frames=len(timeline),
            fps=float(fps),
        )

    # ------------------------------------------------------ motion refinement
    @staticmethod
    def _refine_with_motion(timeline: List[Dict], window: int = 5) -> None:
        """Separate Walking from Standing using measured motion.

        The CNN deliberately sees one frame at a time, so it can only tell
        Walking from Standing by *stance* - legs apart in a stride versus feet
        together. That is the honest limit of a single image. As soon as a
        video is available the real evidence exists, so a person the CNN calls
        "Standing" while their hips are travelling across the room is corrected
        to Walking, and a "Walking" call on someone who has not moved for
        several frames is corrected to Standing.

        Only the two safe upright classes are ever exchanged. Fall decisions
        are never overridden here - emergency logic stays with the classifier
        and the confirmation rule.
        """
        if not timeline:
            return

        speeds = np.array([row.get("hip_speed", 0.0) for row in timeline], dtype=np.float32)
        half = max(window // 2, 1)
        for i, row in enumerate(timeline):
            if row["activity"] not in ("Walking", "Standing"):
                continue
            local = speeds[max(0, i - half): i + half + 1]
            median_speed = float(np.median(local)) if local.size else 0.0

            if row["activity"] == "Standing" and median_speed >= config.MOTION_WALK_SPEED:
                row["activity"] = "Walking"
                row["motion_refined"] = True
            elif row["activity"] == "Walking" and median_speed <= config.MOTION_STATIC_SPEED:
                row["activity"] = "Standing"
                row["motion_refined"] = True
            else:
                row["motion_refined"] = False

    # --------------------------------------------------- temporal confirmation
    @staticmethod
    def _confirm_falls(timeline: List[Dict]):
        """Smooth the per-frame fall probability, then latch confirmed events.

        A single noisy frame must never raise an alarm; a genuine fall stays
        visible for many consecutive frames because the person remains on the
        floor.  Requiring a run of confirmed frames removes almost all of the
        flicker-type false alarms.
        """
        raw = np.array([row["fall_probability"] for row in timeline], dtype=np.float32)
        if len(raw) == 0:
            return [], raw

        window = min(config.SMOOTHING_WINDOW, len(raw))
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(raw, kernel, mode="same")

        alert = smoothed >= config.FALL_PROB_THRESHOLD
        events: List[Dict] = []
        start = None
        for i, flag in enumerate(alert):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                events.append((start, i - 1))
                start = None
        if start is not None:
            events.append((start, len(alert) - 1))

        # Merge runs separated by a short gap - a skeleton lost for a moment
        # while the resident is on the floor must not split one incident in two.
        merged: List[tuple] = []
        for first, last in events:
            if merged and (
                timeline[first]["time"] - timeline[merged[-1][1]]["time"]
                <= config.EVENT_MERGE_GAP_SECONDS
            ):
                merged[-1] = (merged[-1][0], last)
            else:
                merged.append((first, last))
        events = merged

        confirmed: List[Dict] = []
        for first, last in events:
            if last - first + 1 < config.FALL_CONFIRM_FRAMES:
                continue
            segment = timeline[first: last + 1]
            peak = max(segment, key=lambda r: r["fall_probability"])
            lookback = timeline[max(0, first - 4): first + 2]
            max_descent = max((r.get("descent_velocity", 0.0) for r in lookback), default=0.0)
            confirmed.append(
                {
                    "start_time": segment[0]["time"],
                    "end_time": segment[-1]["time"],
                    "duration": round(segment[-1]["time"] - segment[0]["time"], 2),
                    "peak_confidence": peak["fall_probability"],
                    "peak_time": peak["time"],
                    "peak_frame": peak["frame"],
                    "frames": last - first + 1,
                    "rapid_descent": bool(max_descent >= config.RAPID_DESCENT_VELOCITY),
                    "descent_velocity": round(float(max_descent), 3),
                    "severity": (
                        "HIGH"
                        if max_descent >= config.RAPID_DESCENT_VELOCITY
                        or peak["fall_probability"] >= 0.85
                        else "MODERATE"
                    ),
                }
            )
        return confirmed, smoothed

    # ------------------------------------------------------------- analytics
    @staticmethod
    def _summarise(timeline: List[Dict], fall_events: List[Dict], fps: float, elapsed: float) -> Dict:
        detected = [row for row in timeline if row["activity"] != "No person detected"]
        counts = {name: 0 for name in config.CLASS_NAMES}
        for row in detected:
            counts[row["activity"]] = counts.get(row["activity"], 0) + 1

        fall_frames = counts.get(config.FALL_CLASS, 0)
        normal_frames = len(detected) - fall_frames
        confidences = [row["confidence"] for row in detected] or [0.0]

        return {
            "total_frames_analysed": len(timeline),
            "frames_with_person": len(detected),
            "frames_without_person": len(timeline) - len(detected),
            "total_activities_detected": len(detected),
            "activity_counts": counts,
            "fall_frame_count": fall_frames,
            "normal_activity_count": normal_frames,
            "fall_events": len(fall_events),
            "average_confidence": float(np.mean(confidences)),
            "peak_fall_probability": float(
                max((row["fall_probability"] for row in timeline), default=0.0)
            ),
            "dominant_activity": (
                max(counts, key=counts.get) if detected else "No person detected"
            ),
            "detection_rate": len(detected) / max(len(timeline), 1),
            "processing_seconds": round(elapsed, 1),
            "effective_fps": round(len(timeline) / max(elapsed, 1e-3), 1),
            "source_fps": round(float(fps), 1),
            "emergency": len(fall_events) > 0,
        }

    @staticmethod
    def _pick_key_frames(timeline, frames_cache, fall_events, limit: int) -> List[Dict]:
        """Choose the most informative annotated frames to show the caregiver."""
        if not timeline:
            return []

        chosen: List[int] = []
        for event in fall_events:
            for i, row in enumerate(timeline):
                if row["frame"] == event["peak_frame"]:
                    chosen.append(i)
                    break

        remaining = max(limit - len(chosen), 0)
        if remaining:
            candidates = [i for i, r in enumerate(timeline) if r["activity"] != "No person detected"]
            if candidates:
                picks = np.linspace(0, len(candidates) - 1, remaining).round().astype(int)
                chosen.extend(candidates[p] for p in picks)

        seen, key_frames = set(), []
        for i in chosen:
            if i in seen or i >= len(frames_cache):
                continue
            seen.add(i)
            row = timeline[i]
            key_frames.append(
                {
                    "image": frames_cache[i],
                    "time": row["time"],
                    "activity": row["activity"],
                    "confidence": row["confidence"],
                    "is_fall": row["activity"] == config.FALL_CLASS,
                }
            )
        key_frames.sort(key=lambda k: k["time"])
        return key_frames

    def close(self) -> None:
        with self._lock:
            for estimator in (self._pose_video, self._pose_image):
                if estimator is not None:
                    estimator.close()
            self._pose_video = self._pose_image = None
