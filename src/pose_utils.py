"""
SafeFall AI - Pose estimation and feature engineering.

This module is the single source of truth for turning a picture of a person
into the numbers the classifier consumes.  Training and the deployed Streamlit
app both import from here, which guarantees the features seen at inference time
are produced by exactly the same code that produced the training set.

Pipeline
--------
    frame (BGR) -> MediaPipe Pose -> 33 landmarks (x, y, z, visibility)
                -> branch A: hip-centred, torso-scaled landmark tensor (33, 4)
                -> branch B: 25 interpretable geometric features
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np

from . import config

# --------------------------------------------------------------------------- #
# MediaPipe landmark indices (BlazePose full-body topology)
# --------------------------------------------------------------------------- #
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _angle_from_vertical(dx: float, dy: float) -> float:
    """Angle (degrees) between a 2-D vector and the image's vertical axis.

    0 deg  -> perfectly upright,   90 deg -> perfectly horizontal.
    Sign is discarded because leaning left and leaning right are equivalent
    for posture analysis.
    """
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + _EPS)))


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at joint ``b`` formed by the points a-b-c, in degrees."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + _EPS
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _midpoint(lm: np.ndarray, i: int, j: int) -> np.ndarray:
    return (lm[i, :3] + lm[j, :3]) / 2.0


# --------------------------------------------------------------------------- #
# Branch A - normalised landmark tensor
# --------------------------------------------------------------------------- #
def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """Translate to the hip centre and scale by torso length.

    Removes the influence of *where* the person stands in the frame and *how
    far* they are from the camera, so the CNN learns body configuration rather
    than screen position.  Absolute position is not lost from the model as a
    whole - it is supplied separately through the geometric feature branch.
    """
    out = lm.astype(np.float32).copy()
    hip_mid = _midpoint(lm, L_HIP, R_HIP)
    shoulder_mid = _midpoint(lm, L_SHOULDER, R_SHOULDER)

    torso_len = float(np.linalg.norm(shoulder_mid[:2] - hip_mid[:2]))
    if torso_len < 0.02:                       # degenerate / heavily occluded pose
        span_x = float(np.ptp(lm[:, 0]))
        span_y = float(np.ptp(lm[:, 1]))
        torso_len = max(np.hypot(span_x, span_y) / 3.0, 0.02)

    out[:, 0] = (lm[:, 0] - hip_mid[0]) / torso_len
    out[:, 1] = (lm[:, 1] - hip_mid[1]) / torso_len
    out[:, 2] = lm[:, 2] / torso_len
    out[:, 3] = lm[:, 3]                        # visibility passes through
    np.clip(out[:, :3], -6.0, 6.0, out=out[:, :3])
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Branch B - interpretable geometric features
# --------------------------------------------------------------------------- #
# Physically meaningful range for every descriptor. Values outside these bounds
# only ever come from a broken skeleton, so clipping keeps one bad frame from
# dominating the feature scaler.
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "torso_angle": (0.0, 90.0),
    "body_axis_angle": (0.0, 90.0),
    "bbox_aspect_ratio": (0.0, 10.0),
    "bbox_height": (0.0, 2.0),
    "bbox_width": (0.0, 2.0),
    "hip_y": (-1.0, 2.0),
    "shoulder_y": (-1.0, 2.0),
    "ankle_y": (-1.0, 2.0),
    "head_y": (-1.0, 2.0),
    "center_of_mass_y": (-1.0, 2.0),
    "hip_to_ankle_ratio": (-10.0, 10.0),
    "head_above_hip": (-5.0, 5.0),
    "left_knee_angle": (0.0, 180.0),
    "right_knee_angle": (0.0, 180.0),
    "mean_knee_angle": (0.0, 180.0),
    "left_hip_angle": (0.0, 180.0),
    "right_hip_angle": (0.0, 180.0),
    "mean_hip_angle": (0.0, 180.0),
    "ankle_separation": (0.0, 8.0),
    "knee_separation": (0.0, 8.0),
    "shoulder_hip_width_ratio": (0.0, 8.0),
    "leg_asymmetry": (0.0, 8.0),
    "arm_asymmetry": (0.0, 8.0),
    "vertical_horizontal_ratio": (0.0, 12.0),
    "mean_visibility": (0.0, 1.0),
}


def compute_geometric_features(lm: np.ndarray) -> Dict[str, float]:
    """Derive the 25 posture descriptors listed in ``config.GEOMETRIC_FEATURE_NAMES``."""
    shoulder_mid = _midpoint(lm, L_SHOULDER, R_SHOULDER)
    hip_mid = _midpoint(lm, L_HIP, R_HIP)
    knee_mid = _midpoint(lm, L_KNEE, R_KNEE)
    ankle_mid = _midpoint(lm, L_ANKLE, R_ANKLE)

    torso_len = max(float(np.linalg.norm(shoulder_mid[:2] - hip_mid[:2])), _EPS)
    shoulder_width = max(
        float(abs(lm[L_SHOULDER, 0] - lm[R_SHOULDER, 0])), 0.02
    )

    xs, ys = lm[:, 0], lm[:, 1]
    bbox_w = float(np.ptp(xs))
    bbox_h = float(np.ptp(ys))
    aspect = bbox_w / max(bbox_h, _EPS)

    torso_angle = _angle_from_vertical(
        shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]
    )
    body_axis_angle = _angle_from_vertical(
        shoulder_mid[0] - ankle_mid[0], shoulder_mid[1] - ankle_mid[1]
    )

    l_knee = _joint_angle(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE])
    r_knee = _joint_angle(lm[R_HIP], lm[R_KNEE], lm[R_ANKLE])
    l_hip = _joint_angle(lm[L_SHOULDER], lm[L_HIP], lm[L_KNEE])
    r_hip = _joint_angle(lm[R_SHOULDER], lm[R_HIP], lm[R_KNEE])

    feats = {
        "torso_angle": torso_angle,
        "body_axis_angle": body_axis_angle,
        "bbox_aspect_ratio": aspect,
        "bbox_height": bbox_h,
        "bbox_width": bbox_w,
        "hip_y": float(hip_mid[1]),
        "shoulder_y": float(shoulder_mid[1]),
        "ankle_y": float(ankle_mid[1]),
        "head_y": float(lm[NOSE, 1]),
        "center_of_mass_y": float(np.mean(ys)),
        "hip_to_ankle_ratio": float(ankle_mid[1] - hip_mid[1]) / torso_len,
        "head_above_hip": float(hip_mid[1] - lm[NOSE, 1]) / max(bbox_h, _EPS),
        "left_knee_angle": l_knee,
        "right_knee_angle": r_knee,
        "mean_knee_angle": (l_knee + r_knee) / 2.0,
        "left_hip_angle": l_hip,
        "right_hip_angle": r_hip,
        "mean_hip_angle": (l_hip + r_hip) / 2.0,
        "ankle_separation": float(abs(lm[L_ANKLE, 0] - lm[R_ANKLE, 0])) / shoulder_width,
        "knee_separation": float(abs(lm[L_KNEE, 0] - lm[R_KNEE, 0])) / shoulder_width,
        "shoulder_hip_width_ratio": shoulder_width
        / max(float(abs(lm[L_HIP, 0] - lm[R_HIP, 0])), 0.02),
        "leg_asymmetry": float(abs(lm[L_ANKLE, 1] - lm[R_ANKLE, 1])) / torso_len,
        "arm_asymmetry": float(abs(lm[L_WRIST, 1] - lm[R_WRIST, 1])) / torso_len,
        "vertical_horizontal_ratio": bbox_h / max(bbox_w, _EPS),
        "mean_visibility": float(np.mean(lm[:, 3])),
    }

    # Guard against inf/NaN and absurd ratios leaking in from a degenerate pose.
    # Bounds are per-feature: angles are degrees (0-180), the rest are ratios or
    # normalised image coordinates, so a single global clip would silently
    # flatten the joint angles.
    for key, value in feats.items():
        low, high = FEATURE_BOUNDS[key]
        feats[key] = 0.0 if not np.isfinite(value) else float(np.clip(value, low, high))
    return feats


def geometric_feature_vector(lm: np.ndarray) -> np.ndarray:
    """Geometric features as a vector ordered exactly like the config list."""
    feats = compute_geometric_features(lm)
    return np.array(
        [feats[name] for name in config.GEOMETRIC_FEATURE_NAMES], dtype=np.float32
    )


def build_model_inputs(lm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(landmark_tensor (33, 4), geometric_vector (25,))`` for one pose."""
    return normalize_landmarks(lm), geometric_feature_vector(lm)


# --------------------------------------------------------------------------- #
# MediaPipe wrapper
# --------------------------------------------------------------------------- #
class PoseEstimator:
    """Thin, reusable wrapper around MediaPipe Pose.

    MediaPipe objects are stateful and not thread-safe, so one estimator is
    created per worker process (dataset build) or cached once per Streamlit
    session (dashboard).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = config.POSE_MODEL_COMPLEXITY,
        min_detection_confidence: float = config.POSE_MIN_DETECTION_CONF,
        min_tracking_confidence: float = config.POSE_MIN_TRACKING_CONF,
    ) -> None:
        import mediapipe as mp  # imported lazily so the module stays importable

        # MediaPipe graphs are NOT thread-safe. Streamlit starts each script run
        # on a fresh thread, so a live-camera loop and the rerun that replaces it
        # can both be inside this object for a moment. Without a lock that races
        # inside native code and takes the whole process down with no traceback.
        self._lock = threading.RLock()
        self._closed = False
        self._mp = mp
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._pose_module = mp.solutions.pose
        self.pose = self._pose_module.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=not static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # -- core ------------------------------------------------------------- #
    def process(self, frame_bgr: np.ndarray):
        """Run pose estimation on a BGR frame and return the raw MediaPipe result."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        with self._lock:
            if self._closed:
                return None
            return self.pose.process(rgb)

    @staticmethod
    def landmarks_to_array(results) -> Optional[np.ndarray]:
        """Convert a MediaPipe result into a ``(33, 4)`` float32 array."""
        if results is None or not results.pose_landmarks:
            return None
        return np.array(
            [
                [p.x, p.y, p.z, p.visibility]
                for p in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Convenience: frame in, ``(33, 4)`` landmark array (or ``None``) out."""
        return self.landmarks_to_array(self.process(frame_bgr))

    # -- visualisation ----------------------------------------------------- #
    def draw(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Return a copy of the frame with the skeleton overlaid."""
        annotated = frame_bgr.copy()
        if results is not None and results.pose_landmarks:
            with self._lock:
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
        # Take the lock so the graph is never torn down while another thread is
        # still executing inside it.
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.pose.close()
            except Exception:
                pass

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def pose_is_usable(lm: Optional[np.ndarray]) -> bool:
    """Reject frames where the skeleton is too uncertain to be informative."""
    if lm is None:
        return False
    core = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    if float(np.mean(lm[:, 3])) < config.MIN_MEAN_VISIBILITY:
        return False
    if float(np.mean(lm[core, 3])) < 0.5:
        return False
    return True


# --------------------------------------------------------------------------- #
# Framing quality (inference-time only)
# --------------------------------------------------------------------------- #
# The classifier was trained on wall-mounted camera footage in which the whole
# body is visible. A laptop webcam at desk distance shows head and shoulders
# only - and MediaPipe does not simply omit the missing joints, it *predicts*
# where they would be, off the bottom of the frame, with near-zero visibility.
#
# Those invented coordinates flow straight into trunk angle, knee angle and
# stance width, so the model returns a confident answer computed from data that
# does not exist. Refusing to classify is the correct behaviour: a monitor that
# says "I cannot see the resident's legs" is useful, one that guesses is not.
#
# This check is deliberately separate from ``pose_is_usable`` so the training
# and evaluation pipelines are untouched by it.
LOWER_BODY_MIN_VISIBILITY = 0.45
HIP_MIN_VISIBILITY = 0.55
IN_FRAME_MARGIN = 1.04          # normalised y above which a joint is off-screen


class FramingReport:
    """Whether a frame is inside the domain the model was trained on."""

    __slots__ = ("ok", "reason", "advice", "score", "detail")

    def __init__(self, ok: bool, reason: str, advice: str, score: float, detail: dict):
        self.ok = ok
        self.reason = reason
        self.advice = advice
        self.score = score
        self.detail = detail

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FramingReport(ok={self.ok}, reason={self.reason!r}, score={self.score:.2f})"


def check_framing(lm: Optional[np.ndarray]) -> FramingReport:
    """Decide whether the whole body is genuinely in shot."""
    if lm is None:
        return FramingReport(
            False, "no-person",
            "No person detected. Step into the camera's view.", 0.0, {},
        )

    ankle_vis = float(np.mean(lm[[L_ANKLE, R_ANKLE], 3]))
    knee_vis = float(np.mean(lm[[L_KNEE, R_KNEE], 3]))
    hip_vis = float(np.mean(lm[[L_HIP, R_HIP], 3]))
    shoulder_vis = float(np.mean(lm[[L_SHOULDER, R_SHOULDER], 3]))
    ankle_y = float(np.mean(lm[[L_ANKLE, R_ANKLE], 1]))

    detail = {
        "shoulder_visibility": round(shoulder_vis, 3),
        "hip_visibility": round(hip_vis, 3),
        "knee_visibility": round(knee_vis, 3),
        "ankle_visibility": round(ankle_vis, 3),
        "ankle_y": round(ankle_y, 3),
    }
    # Confidence that the full body is in shot, for the live framing meter.
    score = float(np.clip(min(ankle_vis, knee_vis, hip_vis) / LOWER_BODY_MIN_VISIBILITY,
                          0.0, 1.0))

    if shoulder_vis < 0.5:
        return FramingReport(
            False, "no-upper-body",
            "Upper body is not clearly visible. Face the camera.", score, detail,
        )
    if hip_vis < HIP_MIN_VISIBILITY:
        return FramingReport(
            False, "hips-hidden",
            "Move back - your hips need to be in shot. The system needs your "
            "whole body, roughly 2-3 metres from the camera.", score, detail,
        )
    if ankle_vis < LOWER_BODY_MIN_VISIBILITY or knee_vis < LOWER_BODY_MIN_VISIBILITY:
        return FramingReport(
            False, "legs-hidden",
            "Move back - your legs are out of shot. The model reads posture from "
            "hips, knees and ankles, so it needs your whole body in frame.",
            score, detail,
        )
    if ankle_y > IN_FRAME_MARGIN:
        return FramingReport(
            False, "feet-below-frame",
            "Your feet are below the bottom of the frame. Move back or tilt the "
            "camera down.", score, detail,
        )

    return FramingReport(True, "ok", "Full body in shot.", score, detail)


# --------------------------------------------------------------------------- #
# Upper-body fall assessment
# --------------------------------------------------------------------------- #
# The trained CNN needs hips, knees and ankles, so it cannot be asked about a
# desk-distance webcam that only sees head and shoulders. Rather than refuse to
# answer, fall back to cues that ARE reliably measured from the upper body:
#
#   * head axis  - the shoulder-midpoint -> nose vector. Standing, the head sits
#                  directly above the shoulders (~0 deg from vertical). On the
#                  floor it swings out sideways towards 90 deg.
#   * shoulder line tilt - roughly level when upright, steeply tilted when the
#                  person has gone down on their side.
#   * head drop  - a fast downward movement of the head is the signature of the
#                  fall itself, as opposed to lying down deliberately.
#
# This is a geometric detector, not the CNN, and it only distinguishes "on the
# floor" from "upright" - it cannot tell walking from standing. The dashboard
# labels which engine produced each answer so the two are never confused.
# Thresholds measured, not guessed. src/upper_body_eval.py crops 800 labelled
# Le2i frames to a head-and-shoulders view and sweeps each cue:
#
#   shoulder tilt   >= 26 deg   validation accuracy 0.846   <- chosen
#   head axis angle >= 66 deg   validation accuracy 0.717
#
# The head-axis cue was clearly the weaker of the two and is not used as a
# trigger: in a tight crop the nose and shoulder midpoint sit at nearly the same
# height, so the angle saturates near 90 deg for upright people too.
#
# On the held-out test crops the chosen rule gives accuracy 0.785, fall recall
# 0.904 and a 0.324 false-alarm rate. That false-alarm rate is high - it is the
# price of judging a fall without seeing the legs - but recall is what matters
# for a fall monitor, and the dashboard states which engine answered.
UPPER_SHOULDER_TILT_FALL_DEG = 26.0
UPPER_HEAD_DROP_PER_SEC = 0.45


def upper_body_metrics(lm: np.ndarray, aspect: float = 4.0 / 3.0) -> Dict[str, float]:
    """Geometry that survives when only head and shoulders are in shot.

    ``aspect`` is the frame's width/height. MediaPipe returns coordinates
    normalised independently by width and height, so on a non-square frame an
    x-distance and a y-distance of the same numeric size are *different*
    physical distances. Any angle computed straight from those numbers is
    skewed - which on a wide crop was enough to turn an upright person's
    shoulder line into an apparent 45-degree tilt and raise a false fall alert.
    Scaling x back by the aspect ratio recovers true geometric angles.
    """
    shoulder_mid = _midpoint(lm, L_SHOULDER, R_SHOULDER)
    nose = lm[NOSE, :3]

    head_axis_angle = _angle_from_vertical(
        (nose[0] - shoulder_mid[0]) * aspect, nose[1] - shoulder_mid[1]
    )
    dx = float(lm[L_SHOULDER, 0] - lm[R_SHOULDER, 0]) * aspect
    dy = float(lm[L_SHOULDER, 1] - lm[R_SHOULDER, 1])
    shoulder_tilt = float(np.degrees(np.arctan2(abs(dy), abs(dx) + _EPS)))

    return {
        "head_axis_angle": head_axis_angle,
        "shoulder_tilt": shoulder_tilt,
        "head_y": float(nose[1]),
        "shoulder_y": float(shoulder_mid[1]),
        "shoulder_visibility": float(np.mean(lm[[L_SHOULDER, R_SHOULDER], 3])),
    }


def upper_body_fall_score(lm: np.ndarray, head_drop_per_sec: float = 0.0,
                          aspect: float = 4.0 / 3.0) -> tuple:
    """Return ``(is_fall, confidence, metrics)`` from upper-body geometry alone."""
    m = upper_body_metrics(lm, aspect)

    # Shoulder tilt is the validated cue; a rapid head drop is kept as a second
    # trigger because it catches the fall *while it happens*, before the person
    # has settled into a tilted posture.
    tilt_evidence = float(np.clip(m["shoulder_tilt"] / UPPER_SHOULDER_TILT_FALL_DEG, 0, 1.4))
    drop_evidence = float(np.clip(head_drop_per_sec / UPPER_HEAD_DROP_PER_SEC, 0, 1.4))

    score = max(tilt_evidence, drop_evidence)
    is_fall = score >= 1.0
    # Map the evidence onto a readable confidence either side of the decision.
    confidence = float(np.clip(0.5 + (score - 1.0) * 0.5, 0.05, 0.99)) if is_fall \
        else float(np.clip(0.5 + (1.0 - score) * 0.5, 0.05, 0.99))

    m.update({
        "head_drop_per_sec": round(head_drop_per_sec, 3),
        "evidence": round(score, 3),
    })
    return is_fall, confidence, m
