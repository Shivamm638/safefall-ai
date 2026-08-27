"""
SafeFall AI - live camera that does not need WebRTC.

Why this exists
---------------
The WebRTC page streams video as a peer-to-peer media connection, which needs
the browser and the server to find a route to each other. On a deployed host
they cannot: the Streamlit Cloud container sits behind NAT that will not accept
an inbound media connection, so the two sides need a TURN relay to forward the
media - and every TURN provider requires an account, because relaying costs
bandwidth. That makes real-time monitoring on the public link depend on a
credential the visitor may not have.

This module removes that dependency. The browser captures frames itself and
hands them to the Python script through Streamlit's *own* component channel -
the same websocket that already carries every widget value and rerun, and which
demonstrably works on the deployed app. No STUN, no TURN, no configuration, and
it works for any visitor who opens the link.

The trade-off is honest: each frame costs a script rerun, so the rate is
whatever the round-trip allows - a few frames a second rather than the 15-30
of a true media stream. For fall detection that is sufficient, because the
alarm confirms over consecutive frames rather than reacting to a single one.
The WebRTC page remains the better option wherever a relay is available.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit.components.v1 as components

COMPONENT_DIR = Path(__file__).resolve().parent.parent / "components" / "browser_camera"

_component = None


def _get_component():
    """Declare the component once per process.

    ``declare_component`` registers a route on Streamlit's static server, so it
    must not be called on every rerun.
    """
    global _component
    if _component is None:
        if not (COMPONENT_DIR / "index.html").exists():
            raise FileNotFoundError(
                f"Browser camera component missing at {COMPONENT_DIR}. "
                "It ships with the repository; check that components/ was committed."
            )
        _component = components.declare_component(
            "safefall_browser_camera", path=str(COMPONENT_DIR)
        )
    return _component


def browser_camera(
    running: bool,
    width: int = 480,
    quality: float = 0.6,
    mirror: bool = True,
    min_interval_ms: int = 120,
    ack: int = 0,
    key: str = "safefall_browser_camera",
):
    """Render the camera and return the most recent payload from the browser.

    The return value is ``None`` until the first frame arrives, then a dict of
    either ``{"frame": <data URI>, "seq": int, ...}`` or ``{"error": str}``.
    ``seq`` increments per frame and is how the caller tells a genuinely new
    frame from a rerun that merely re-delivered the last one.

    ``ack`` must be the ``seq`` the caller has finished analysing. It is what
    drives the loop: Streamlit only re-renders a component when its arguments
    change, so a constant argument list means the component is never asked for
    another frame and the stream crawls along on its safety-net timer. Feeding
    the sequence number back both changes the arguments every frame and keeps
    exactly one frame in flight.
    """
    return _get_component()(
        running=bool(running),
        width=int(width),
        quality=float(quality),
        mirror=bool(mirror),
        min_interval_ms=int(min_interval_ms),
        ack=int(ack),
        key=key,
        default=None,
    )


def decode_frame(data_url: str) -> Optional[np.ndarray]:
    """Turn the browser's JPEG data URI into the BGR array the pipeline expects."""
    if not data_url or "," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
    except Exception:  # noqa: BLE001 - a truncated payload must not kill the page
        return None
    buffer = np.frombuffer(raw, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
