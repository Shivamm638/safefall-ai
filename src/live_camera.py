"""
SafeFall AI - live webcam monitoring.

Reads a camera with OpenCV and keeps the rolling state the dashboard needs to
show a continuous read-out: current activity, confidence, a smoothed fall
probability, and a latched alarm that only fires after several consecutive
confirmed frames (the same principle as the uploaded-video pipeline).

Where this runs
---------------
OpenCV opens the camera on whichever machine is running the Python process, so
continuous monitoring works when the dashboard runs on the same computer as the
webcam. A cloud host has no camera attached, so ``camera_available()`` returns
False there and the dashboard falls back to single-snapshot capture, which
happens in the visitor's own browser and works anywhere.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from . import config


def _backend():
    """Prefer DirectShow on Windows - the default MSMF backend is slow to open."""
    import cv2

    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def open_camera(index: int = 0, width: int = config.LIVE_DEFAULT_WIDTH,
                height: int = config.LIVE_DEFAULT_HEIGHT,
                warmup_seconds: float = 3.0):
    """Open a camera and return the capture, or ``None`` if it cannot be used."""
    import cv2

    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # always grab the freshest frame

    # A DirectShow webcam does not deliver at full rate straight away. While its
    # auto-exposure settles - a few seconds, longer in a dim room - it runs at
    # roughly 1 FPS and individual reads can fail outright. Measured on this
    # machine: ~1 FPS for the first five seconds, then a steady ~27 FPS.
    #
    # So the open must be patient. Discarding a handful of warm-up frames here
    # means the monitoring loop starts on a camera that is already up to speed,
    # instead of mistaking the warm-up for a dead device.
    if not warm_up(cap, seconds=warmup_seconds):
        cap.release()
        return None
    return cap


def warm_up(cap, seconds: float = 3.0, min_good_frames: int = 3) -> bool:
    """Read and discard frames until the camera is delivering reliably."""
    import time

    deadline = time.time() + seconds
    good = 0
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            good += 1
            if good >= min_good_frames:
                return True
        else:
            time.sleep(0.05)
    return good > 0


def camera_available(index: int = 0) -> bool:
    """Cheap probe used to decide between live streaming and snapshot mode.

    Deliberately does *not* wait for the full warm-up: this only answers "is
    there a camera here at all", and every extra open/release cycle costs the
    next opener another settling period.
    """
    try:
        cap = open_camera(index, warmup_seconds=1.0)
    except Exception:
        return False
    if cap is None:
        return False
    cap.release()
    return True


def list_cameras(max_index: int = 3) -> List[int]:
    """Indices of cameras that actually deliver a frame."""
    found = []
    for index in range(max_index):
        if camera_available(index):
            found.append(index)
    return found


@dataclass
class LiveMonitorState:
    """Rolling statistics for a live monitoring session."""

    started_at: float = field(default_factory=time.time)
    frames: int = 0
    frames_with_person: int = 0
    activity_counts: Dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in config.CLASS_NAMES}
    )
    confidences: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.LIVE_HISTORY_FRAMES)
    )
    fall_probs: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.LIVE_HISTORY_FRAMES)
    )
    activities: Deque[str] = field(
        default_factory=lambda: deque(maxlen=config.LIVE_HISTORY_FRAMES)
    )
    timestamps: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.LIVE_HISTORY_FRAMES)
    )
    frames_out_of_frame: int = 0
    consecutive_fall_frames: int = 0
    alarm_active: bool = False
    fall_events: List[Dict] = field(default_factory=list)
    _frame_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=30), repr=False
    )

    # ------------------------------------------------------------------ update
    def update(self, prediction, now: Optional[float] = None) -> bool:
        """Fold one frame into the session. Returns True if the alarm just fired."""
        now = time.time() if now is None else now
        self.frames += 1
        self._frame_times.append(now)

        # Upper-body observations now count: the system answers in that mode
        # too, just with a different engine. Only a genuinely absent person is
        # excluded from the statistics.
        if not prediction.framing_ok:
            self.frames_out_of_frame += 1
        if not prediction.pose_found:
            self.activities.append("No person")
            self.confidences.append(0.0)
            self.fall_probs.append(0.0)
            self.timestamps.append(now - self.started_at)
            # Losing the skeleton must not silently clear a live alarm, but it
            # should not build one either.
            self.consecutive_fall_frames = max(self.consecutive_fall_frames - 1, 0)
            return False

        self.frames_with_person += 1
        self.activity_counts[prediction.activity] = (
            self.activity_counts.get(prediction.activity, 0) + 1
        )
        fall_probability = prediction.probabilities[config.FALL_CLASS]

        self.activities.append(prediction.activity)
        self.confidences.append(float(prediction.confidence))
        self.fall_probs.append(float(fall_probability))
        self.timestamps.append(now - self.started_at)

        if fall_probability >= config.FALL_PROB_THRESHOLD:
            self.consecutive_fall_frames += 1
        else:
            self.consecutive_fall_frames = 0

        just_fired = False
        if self.consecutive_fall_frames >= config.LIVE_CONFIRM_FRAMES:
            if not self.alarm_active:
                just_fired = True
                self.fall_events.append(
                    {
                        "at_seconds": round(now - self.started_at, 1),
                        "confidence": round(float(fall_probability), 4),
                        "frame": self.frames,
                    }
                )
            self.alarm_active = True
        elif self.consecutive_fall_frames == 0:
            self.alarm_active = False

        return just_fired

    # ----------------------------------------------------------------- read-out
    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    @property
    def rolling_confidence(self) -> float:
        """Mean confidence over the recent window - how settled the model is."""
        window = [c for c in list(self.confidences)[-config.LIVE_ROLLING_WINDOW:] if c > 0]
        return float(np.mean(window)) if window else 0.0

    @property
    def smoothed_fall_probability(self) -> float:
        window = list(self.fall_probs)[-config.SMOOTHING_WINDOW:]
        return float(np.mean(window)) if window else 0.0

    @property
    def detection_rate(self) -> float:
        return self.frames_with_person / self.frames if self.frames else 0.0

    @property
    def usable_rate(self) -> float:
        """Share of frames that were actually classifiable (whole body in shot)."""
        return self.frames_with_person / self.frames if self.frames else 0.0

    @property
    def dominant_activity(self) -> str:
        counts = {k: v for k, v in self.activity_counts.items() if v}
        return max(counts, key=counts.get) if counts else "No person"

    @property
    def fall_frame_count(self) -> int:
        return self.activity_counts.get(config.FALL_CLASS, 0)

    @property
    def safe_frame_count(self) -> int:
        return self.frames_with_person - self.fall_frame_count

    @property
    def stability(self) -> float:
        """Share of the recent window agreeing with the current activity.

        High stability means the model is holding a steady opinion rather than
        flickering between classes - the live equivalent of a confident call.
        """
        recent = [a for a in list(self.activities)[-config.LIVE_ROLLING_WINDOW:]
                  if a not in ("No person", "Out of frame")]
        if not recent:
            return 0.0
        return recent.count(recent[-1]) / len(recent)

    def history_frame(self):
        """Rolling history as a DataFrame for the live charts."""
        import pandas as pd

        return pd.DataFrame(
            {
                "seconds": list(self.timestamps),
                "activity": list(self.activities),
                "confidence": list(self.confidences),
                "fall_probability": list(self.fall_probs),
            }
        )


# --------------------------------------------------------------------------- #
# Network cameras (CCTV / IP / phone cameras)
# --------------------------------------------------------------------------- #
# OpenCV can read an RTSP or MJPEG stream exactly like a local device, so the
# same pipeline serves a wall-mounted CCTV camera - which is the deployment this
# model was actually trained for, since Le2i footage comes from fixed cameras
# viewing a whole room.
#
# Two practical hazards are handled here:
#   * a bad URL makes OpenCV block forever, so open/read timeouts are set;
#   * RTSP URLs usually embed a password, so nothing here ever logs the raw URL
#     and ``mask_credentials`` is used for anything shown on screen.

STREAM_OPEN_TIMEOUT_MS = 6000
STREAM_READ_TIMEOUT_MS = 6000


def mask_credentials(url: str) -> str:
    """Hide any ``user:password@`` section before a URL is displayed."""
    if "@" not in url or "//" not in url:
        return url
    scheme, _, rest = url.partition("//")
    creds, _, host = rest.rpartition("@")
    if not creds:
        return url
    user = creds.split(":", 1)[0]
    return f"{scheme}//{user}:****@{host}"


def _blocking_open(url: str, result: dict) -> None:
    """Open a capture and hand it back through ``result`` (runs in a worker)."""
    import cv2

    cap = None
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            result["cap"] = None
            return
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok, _ = cap.read()
        if not ok:
            cap.release()
            result["cap"] = None
            return
        result["cap"] = cap
    except Exception as exc:  # noqa: BLE001
        if cap is not None:
            cap.release()
        result["cap"] = None
        result["error"] = str(exc)
    finally:
        # If the caller already gave up, nobody will ever release this capture.
        if result.get("abandoned") and result.get("cap") is not None:
            result["cap"].release()
            result["cap"] = None


def open_stream(url: str, timeout_ms: int = STREAM_OPEN_TIMEOUT_MS):
    """Open a network camera stream, or return ``None`` if it cannot be read.

    The timeout is enforced here in Python rather than left to FFmpeg. FFmpeg's
    own ``timeout``/``stimeout`` options are not honoured by every OpenCV build -
    on this one an unreachable RTSP host still blocked for its default 30 s,
    which means a mistyped URL leaves the dashboard frozen on a spinner. Opening
    in a worker thread and abandoning it after the budget gives a predictable
    few-second failure on any backend.
    """
    import os
    import threading

    # Still ask FFmpeg nicely - it costs nothing where it *is* supported, and it
    # forces TCP for RTSP (UDP silently drops frames on congested wifi and
    # produces a smeared picture rather than a clean failure).
    micros = int(timeout_ms) * 1000
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;tcp|timeout;{micros}|stimeout;{micros}"
    )

    result: dict = {}
    worker = threading.Thread(target=_blocking_open, args=(url, result), daemon=True)
    worker.start()
    worker.join(timeout=timeout_ms / 1000.0)

    if worker.is_alive():
        # Let the straggler clean up after itself once it eventually returns.
        result["abandoned"] = True
        return None
    return result.get("cap")


def test_stream(url: str):
    """Try a stream once. Returns ``(ok, message, first_frame_or_None)``."""
    safe = mask_credentials(url)
    if not url.strip():
        return False, "Enter a camera URL first.", None
    try:
        cap = open_stream(url)
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not open {safe}: {exc}", None
    if cap is None:
        return False, (
            f"No video from {safe}. Check the URL, that the camera is on the same "
            "network, and that the username/password are correct."
        ), None
    ok, frame = cap.read()
    width = int(cap.get(3)) or 0
    height = int(cap.get(4)) or 0
    cap.release()
    if not ok or frame is None:
        return False, f"Connected to {safe} but no frame arrived.", None
    return True, f"Connected to {safe} - {width}x{height}", frame


def open_source(source, width: int = config.LIVE_DEFAULT_WIDTH,
                height: int = config.LIVE_DEFAULT_HEIGHT):
    """Open either a local device index or a network stream URL."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return open_camera(int(source), width, height)
    return open_stream(str(source))


# --------------------------------------------------------------------------- #
# Always-fresh capture
# --------------------------------------------------------------------------- #
class FreshestFrame:
    """Keep only the newest frame, in a background thread.

    Reading a camera synchronously inside the analysis loop makes the *camera*
    run at the analysis rate. Measured here: analysing every frame pulled the
    capture down from 30 FPS to 14.8 FPS, because each ``read()`` returned the
    next queued frame rather than the current one. Every displayed frame was
    therefore already stale, and the delay compounded - which is exactly what
    "it lags as soon as I turn the camera on" feels like.

    This thread reads continuously and throws away anything the consumer did not
    collect in time. The camera stays at its native rate, the loop always gets
    the present moment, and latency stops accumulating no matter how slow the
    analysis is.
    """

    def __init__(self, cap):
        import threading

        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._running = True
        self._failures = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self._failures += 1
                time.sleep(0.02)
                continue
            self._failures = 0
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self, wait_for_new: bool = True, timeout: float = 1.0):
        """Return ``(ok, frame)`` - the most recent frame the thread has seen."""
        deadline = time.time() + timeout
        last_seen = getattr(self, "_last_seq", -1)
        while True:
            with self._lock:
                frame, seq = self._frame, self._seq
            if frame is not None and (not wait_for_new or seq != last_seen):
                self._last_seq = seq
                return True, frame
            if time.time() > deadline:
                return (frame is not None), frame
            time.sleep(0.004)

    @property
    def read_failures(self) -> int:
        return self._failures

    def release(self) -> None:
        self._running = False
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        self._cap.release()
