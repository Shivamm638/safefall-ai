"""
SafeFall AI - real-time monitoring over WebRTC.

Why this exists
---------------
The Laptop Webcam page opens the camera with OpenCV, which reads the camera of
whichever machine runs the Python process. That is the user's own laptop when
the app runs locally - and a datacentre server with no camera once it is
deployed, which is why the deployed app can only offer still snapshots.

WebRTC inverts that. The **viewer's browser** captures the video and streams it
to the server, which processes each frame and streams the annotated result
back. So the deployed app gets genuine real-time monitoring from whatever camera
the visitor has - laptop webcam, phone camera, anything the browser can open.

Threading note: the frame callback runs on a WebRTC worker thread, not
Streamlit's script thread, so nothing here may touch ``st.session_state``.
Shared state goes through a lock-protected object the page polls instead.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from . import config

# Public STUN server, needed so the browser and the Streamlit host can find a
# route to each other through NAT. No media passes through it.
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


class LiveStats:
    """Thread-safe rolling statistics shared between the callback and the page."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.started_at = time.time()
            self.frames = 0
            self.frames_with_person = 0
            self.frames_out_of_frame = 0
            self.activity_counts: Dict[str, int] = {n: 0 for n in config.CLASS_NAMES}
            self.current_activity = "Waiting for camera"
            self.current_confidence = 0.0
            self.current_engine = ""
            self.framing_ok = False
            self.framing_score = 0.0
            self.framing_advice = ""
            self.consecutive_fall_frames = 0
            self.alarm_active = False
            self.fall_events: list = []
            self._recent_conf: Deque[float] = deque(maxlen=config.LIVE_ROLLING_WINDOW)
            self._frame_times: Deque[float] = deque(maxlen=30)

    def update(self, prediction) -> None:
        now = time.time()
        with self._lock:
            self.frames += 1
            self._frame_times.append(now)
            self.current_engine = getattr(prediction, "engine", "")
            self.framing_ok = bool(prediction.framing_ok)
            self.framing_score = float(prediction.framing_score)
            self.framing_advice = prediction.framing_advice

            if not prediction.pose_found:
                self.current_activity = "No person in view"
                self.current_confidence = 0.0
                self.consecutive_fall_frames = max(self.consecutive_fall_frames - 1, 0)
                if self.consecutive_fall_frames == 0:
                    self.alarm_active = False
                return

            self.frames_with_person += 1
            if not prediction.framing_ok:
                self.frames_out_of_frame += 1

            self.current_activity = prediction.activity
            self.current_confidence = float(prediction.confidence)
            self._recent_conf.append(float(prediction.confidence))
            self.activity_counts[prediction.activity] = (
                self.activity_counts.get(prediction.activity, 0) + 1
            )

            fall_p = prediction.probabilities.get(config.FALL_CLASS, 0.0)
            if fall_p >= config.FALL_PROB_THRESHOLD:
                self.consecutive_fall_frames += 1
            else:
                self.consecutive_fall_frames = 0

            if self.consecutive_fall_frames >= config.LIVE_CONFIRM_FRAMES:
                if not self.alarm_active:
                    self.fall_events.append({
                        "at_seconds": round(now - self.started_at, 1),
                        "confidence": round(float(fall_p), 4),
                    })
                self.alarm_active = True
            elif self.consecutive_fall_frames == 0:
                self.alarm_active = False

    def snapshot(self) -> Dict:
        """A consistent copy for the page to render."""
        with self._lock:
            span = (self._frame_times[-1] - self._frame_times[0]
                    if len(self._frame_times) > 1 else 0.0)
            fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            recent = [c for c in self._recent_conf if c > 0]
            return {
                "frames": self.frames,
                "frames_with_person": self.frames_with_person,
                "frames_out_of_frame": self.frames_out_of_frame,
                "activity": self.current_activity,
                "confidence": self.current_confidence,
                "rolling_confidence": float(np.mean(recent)) if recent else 0.0,
                "engine": self.current_engine,
                "framing_ok": self.framing_ok,
                "framing_score": self.framing_score,
                "framing_advice": self.framing_advice,
                "alarm_active": self.alarm_active,
                "fall_events": list(self.fall_events),
                "activity_counts": dict(self.activity_counts),
                "elapsed": time.time() - self.started_at,
                "fps": fps,
            }


def make_frame_callback(predictor, stats: LiveStats, analyse_every: int = 2,
                        mirror: bool = True):
    """Build the per-frame callback WebRTC will invoke on its worker thread.

    ``analyse_every`` decouples display from analysis: every frame is shown so
    the video stays smooth, while the pose model runs on one frame in N. The
    most recent annotated result is reused in between, which keeps the skeleton
    on screen without paying for it on every frame.
    """
    import av
    import cv2

    state = {"count": 0, "last_annotated": None, "last_shape": None}

    def callback(frame: "av.VideoFrame") -> "av.VideoFrame":
        image = frame.to_ndarray(format="bgr24")
        if mirror:
            image = cv2.flip(image, 1)

        state["count"] += 1
        run_model = (state["count"] % max(analyse_every, 1)) == 0

        if run_model:
            try:
                prediction = predictor.predict_image(
                    image, draw=True, static=False, enforce_framing=True
                )
                stats.update(prediction)
                if prediction.annotated_image is not None:
                    state["last_annotated"] = prediction.annotated_image
                    output = prediction.annotated_image
                else:
                    output = image
            except Exception:
                # A bad frame must never kill the stream - show it unannotated.
                output = image
        else:
            last = state["last_annotated"]
            output = last if last is not None and last.shape == image.shape else image

        output = _overlay_status(output, stats.snapshot())
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    return callback


def _overlay_status(image: np.ndarray, snap: Dict) -> np.ndarray:
    """Burn the verdict into the video itself.

    The banner has to travel *in the frame*: the video element is rendered by
    the browser and cannot be overlaid with Streamlit widgets, and the metric
    panel beside it only refreshes when the page reruns.
    """
    import cv2

    out = image
    height, width = out.shape[:2]
    alarm = snap["alarm_active"]
    ok = snap["framing_ok"]

    if alarm:
        colour, label = (60, 60, 255), "FALL DETECTED - EMERGENCY"
    elif not ok:
        colour, label = (64, 178, 245), "MOVE BACK - WHOLE BODY NOT IN SHOT"
    else:
        colour, label = (143, 190, 60), snap["activity"].upper()

    bar = 46
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar), colour, -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (255, 255, 255), 2, cv2.LINE_AA)

    if ok and not alarm:
        cv2.putText(out, f"{snap['confidence']:.0%}", (width - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    # Framing meter along the bottom edge.
    meter_h = 8
    cv2.rectangle(out, (0, height - meter_h), (width, height), (40, 40, 40), -1)
    filled = int(width * max(min(snap["framing_score"], 1.0), 0.0))
    meter_colour = (143, 190, 60) if ok else (64, 178, 245)
    cv2.rectangle(out, (0, height - meter_h), (filled, height), meter_colour, -1)
    if alarm:
        cv2.rectangle(out, (0, 0), (width - 1, height - 1), (60, 60, 255), 4)
    return out
