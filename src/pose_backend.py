"""
SafeFall AI - MediaPipe backend selection.

Why this file exists
--------------------
MediaPipe split into two incompatible eras, and no single release spans them:

    mediapipe <= 0.10.21   has ``mp.solutions.pose``   no wheels above Python 3.12
    mediapipe >= 0.10.30   ``mp.solutions`` REMOVED    wheels for 3.13 / 3.14

The project was built on ``mp.solutions``. Streamlit Community Cloud provisions
the newest interpreter it has (3.14 at time of writing), where only the second
group installs - so pinning the old version fails to build, and pinning a new
one fails at import.

Rather than making the deployment depend on someone remembering to set a Python
version in a web form, this module detects which API is present and adapts. Both
backends drive the *same* BlazePose network and return the same 33 landmarks in
the same order and normalisation, so everything downstream - the feature
engineering, the scaler, the trained CNN - is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import config

# The Tasks API needs the network as a file. "full" is the same complexity-1
# model that mp.solutions used by default, so landmarks match the training data.
TASK_MODEL_PATH = config.MODELS_DIR / "pose_landmarker_full.task"
TASK_MODEL_LITE_PATH = config.MODELS_DIR / "pose_landmarker_lite.task"

# BlazePose skeleton, used for drawing when the legacy drawing helper is absent.
POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


def has_legacy_solutions() -> bool:
    """True when this mediapipe build still exposes ``mp.solutions``."""
    try:
        import mediapipe as mp

        return hasattr(mp, "solutions")
    except Exception:
        return False


class _PoseResult:
    """Minimal stand-in for the legacy result object.

    Only ``pose_landmarks.landmark`` is ever read downstream, so the Tasks
    backend fills that shape rather than the rest of the codebase learning about
    two different result types.
    """

    __slots__ = ("pose_landmarks",)

    def __init__(self, landmarks) -> None:
        self.pose_landmarks = landmarks


class _LandmarkList:
    __slots__ = ("landmark",)

    def __init__(self, landmark) -> None:
        self.landmark = landmark


class _Landmark:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility) -> None:
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class TasksBackend:
    """Pose estimation through ``mediapipe.tasks`` (MediaPipe 0.10.30+)."""

    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = TASK_MODEL_PATH
        if model_complexity == 0 and TASK_MODEL_LITE_PATH.exists():
            model_path = TASK_MODEL_LITE_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. It ships with the repo; "
                "re-download with src/download_pose_model.py if missing."
            )

        self._mp = mp
        self._vision = vision
        self._running_mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=self._running_mode,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONF,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONF,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )

        if self._running_mode == self._vision.RunningMode.VIDEO:
            # Timestamps must increase strictly in VIDEO mode.
            self._timestamp_ms += 33
            result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        else:
            result = self._landmarker.detect(image)

        if not result.pose_landmarks:
            return _PoseResult(None)

        landmarks = result.pose_landmarks[0]
        # The Tasks API moved visibility onto the landmark itself; older builds
        # leave it None, in which case presence of the joint implies confidence.
        converted = [
            _Landmark(
                lm.x, lm.y, lm.z,
                lm.visibility if getattr(lm, "visibility", None) is not None else 1.0,
            )
            for lm in landmarks
        ]
        return _PoseResult(_LandmarkList(converted))

    def draw(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Draw the skeleton to match the legacy overlay's appearance."""
        import cv2

        annotated = frame_bgr.copy()
        if results is None or results.pose_landmarks is None:
            return annotated

        height, width = annotated.shape[:2]
        points = [
            (int(lm.x * width), int(lm.y * height))
            for lm in results.pose_landmarks.landmark
        ]
        for a, b in POSE_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(annotated, points[a], points[b], (255, 200, 0), 2)
        for x, y in points:
            cv2.circle(annotated, (x, y), 3, (0, 255, 170), -1)
        return annotated

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


class LegacyBackend:
    """Pose estimation through ``mp.solutions.pose`` (MediaPipe <= 0.10.21)."""

    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1):
        import mediapipe as mp

        self._mp = mp
        self._drawing = mp.solutions.drawing_utils
        self._pose_module = mp.solutions.pose
        self.pose = self._pose_module.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=not static_image_mode,
            min_detection_confidence=config.POSE_MIN_DETECTION_CONF,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONF,
        )

    def process(self, frame_bgr: np.ndarray):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.pose.process(rgb)

    def draw(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        annotated = frame_bgr.copy()
        if results is not None and results.pose_landmarks:
            self._drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self._pose_module.POSE_CONNECTIONS,
                landmark_drawing_spec=self._drawing.DrawingSpec(
                    color=(0, 255, 170), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=self._drawing.DrawingSpec(
                    color=(255, 200, 0), thickness=2
                ),
            )
        return annotated

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass


def create_backend(static_image_mode: bool = False, model_complexity: int = 1):
    """Return whichever backend this MediaPipe build supports."""
    if has_legacy_solutions():
        return LegacyBackend(static_image_mode, model_complexity)
    return TasksBackend(static_image_mode, model_complexity)


def backend_name() -> str:
    return "mp.solutions (legacy)" if has_legacy_solutions() else "mediapipe.tasks"
