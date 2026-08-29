"""
SafeFall AI - Elderly Fall Detection & Activity Monitoring Dashboard
====================================================================

Streamlit healthcare dashboard for the FA-2 assessment.

    python -m streamlit run app.py

Loads the CNN trained in ``src/train.py`` and runs the complete pipeline -
MediaPipe Pose -> feature engineering -> CNN -> emergency logic - on a live
webcam feed, uploaded photos, uploaded videos and camera snapshots.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.inference import SafeFallPredictor  # noqa: E402
from src.live_camera import (  # noqa: E402
    FreshestFrame,
    LiveMonitorState,
    camera_available,
    mask_credentials,
    open_source,
    test_stream,
)

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="SafeFall AI - Elderly Fall Detection",
    page_icon="\U0001F6E1️",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#E6EDF6"          # primary text on the dark surface
MUTED = "#8CA0B8"
PRIMARY = "#2DD4BF"      # teal accent
ACCENT = "#5AA9F7"
DANGER = "#FF5A5F"
SUCCESS = "#34D399"
GRID = "#22304A"

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      :root {
        --sf-bg:      #0B1220;
        --sf-surface: #141E30;
        --sf-ink:     #E6EDF6;
        --sf-muted:   #8CA0B8;
        --sf-line:    #24334D;
        --sf-primary: #2DD4BF;
        --sf-accent:  #5AA9F7;
        --sf-danger:  #FF5A5F;
        --sf-success: #34D399;
        --sf-radius:  16px;
      }

      html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      .stApp { background: var(--sf-bg); color: var(--sf-ink); }
      .block-container { padding: 1.4rem 2.2rem 4rem; max-width: 1480px; }
      #MainMenu, footer { visibility: hidden; }

      /* ---------------- hero ---------------- */
      .sf-hero {
        background:
          radial-gradient(900px 340px at 6% -50%, rgba(45,212,191,.20), transparent 62%),
          linear-gradient(118deg, #0E1B2E 0%, #14263D 55%, #16324A 100%);
        border: 1px solid var(--sf-line);
        color:#fff; padding:1.35rem 1.7rem; border-radius:var(--sf-radius);
        margin-bottom:1.25rem;
      }
      .sf-hero h1 { margin:0; font-size:1.5rem; font-weight:800; letter-spacing:-.02em;
                    color:#fff; }
      .sf-hero p  { margin:.35rem 0 0; color:var(--sf-muted); font-size:.93rem; max-width:80ch; }
      .sf-tags { margin-top:.85rem; display:flex; gap:.4rem; flex-wrap:wrap; }
      .sf-tag {
        background:rgba(45,212,191,.10); border:1px solid rgba(45,212,191,.28);
        color:var(--sf-primary);
        padding:.2rem .68rem; border-radius:999px; font-size:.72rem; font-weight:600;
      }

      /* ---------------- cards ---------------- */
      .sf-card {
        background:var(--sf-surface); border:1px solid var(--sf-line);
        border-radius:var(--sf-radius); padding:1rem 1.15rem; height:100%;
      }
      .sf-card:hover { border-color:#2F4467; }
      .sf-card .sf-top { display:flex; align-items:center; gap:.45rem; margin-bottom:.28rem; }
      .sf-card .sf-ico { font-size:.92rem; opacity:.9; }
      .sf-card .sf-label {
        color:var(--sf-muted); font-size:.68rem; font-weight:700;
        text-transform:uppercase; letter-spacing:.09em;
      }
      .sf-card .sf-value { font-size:1.8rem; font-weight:800; line-height:1.15;
                           letter-spacing:-.02em; }
      .sf-card .sf-sub { color:var(--sf-muted); font-size:.74rem; margin-top:.18rem; }

      /* ---------------- status banners ---------------- */
      .sf-alert {
        background:linear-gradient(118deg, rgba(255,90,95,.22), rgba(255,90,95,.08));
        border:1px solid rgba(255,90,95,.45); border-left:7px solid var(--sf-danger);
        padding:1.2rem 1.5rem; border-radius:var(--sf-radius);
        margin:.45rem 0 1.05rem; animation:sfpulse 1.5s ease-in-out infinite;
      }
      .sf-alert h2 { margin:0; font-size:1.32rem; font-weight:800; color:#FFB3B5; }
      .sf-alert p  { margin:.35rem 0 0; font-size:.93rem; color:#F2D6D7; }
      @keyframes sfpulse {
        0%,100% { box-shadow:0 0 0 0 rgba(255,90,95,.30); }
        50%     { box-shadow:0 0 28px 2px rgba(255,90,95,.42); }
      }

      .sf-safe {
        background:linear-gradient(118deg, rgba(52,211,153,.16), rgba(52,211,153,.05));
        border:1px solid rgba(52,211,153,.38); border-left:7px solid var(--sf-success);
        padding:1.05rem 1.45rem; border-radius:var(--sf-radius); margin:.45rem 0 1.05rem;
      }
      .sf-safe h2 { margin:0; font-size:1.18rem; font-weight:800; color:#7FE9C4; }
      .sf-safe p  { margin:.28rem 0 0; font-size:.9rem; color:#C6EEDF; }

      .sf-warn {
        background:rgba(245,178,64,.12); border:1px solid rgba(245,178,64,.40);
        border-left:7px solid #F5B240; color:#FBE3B8;
        padding:1rem 1.3rem; border-radius:12px; margin:.45rem 0 1.05rem;
      }
      .sf-warn b { color:#FFD37A; }
      .sf-info {
        background:rgba(90,169,247,.10); border:1px solid rgba(90,169,247,.34);
        border-left:7px solid var(--sf-accent); color:#CFE3FA;
        padding:1rem 1.3rem; border-radius:12px; margin:.45rem 0 1.05rem; font-size:.9rem;
      }

      /* ---------------- section headings ---------------- */
      .sf-section {
        display:flex; align-items:center; gap:.55rem;
        font-size:1rem; font-weight:750; color:var(--sf-ink);
        margin:1.5rem 0 .7rem; letter-spacing:-.01em;
      }
      .sf-section::before {
        content:""; width:4px; height:1rem; border-radius:3px;
        background:linear-gradient(180deg, var(--sf-primary), var(--sf-accent));
      }
      .sf-note { color:var(--sf-muted); font-size:.84rem; line-height:1.55; }

      /* ---------------- probability bars ---------------- */
      .sf-bars { display:flex; flex-direction:column; gap:.5rem; }
      .sf-bar-row .sf-bar-top {
        display:flex; justify-content:space-between; font-size:.8rem;
        font-weight:600; color:var(--sf-ink); margin-bottom:.2rem;
      }
      .sf-bar-track { background:#1C2A42; border-radius:999px; height:9px; overflow:hidden; }
      .sf-bar-fill { height:100%; border-radius:999px; transition:width .18s ease; }

      /* ---------------- live feed + framing meter ---------------- */
      .sf-feed {
        width:100%; display:block; border-radius:var(--sf-radius);
        border:1px solid var(--sf-line); background:#060B14;
      }
      .sf-frame-meter { margin:.6rem 0 .1rem; }
      .sf-frame-meter .t {
        display:flex; justify-content:space-between; font-size:.75rem;
        font-weight:700; letter-spacing:.03em; margin-bottom:.24rem;
      }
      .sf-frame-track { background:#1C2A42; border-radius:999px; height:7px; overflow:hidden; }
      .sf-frame-fill { height:100%; border-radius:999px; transition:width .2s ease; }

      /* ---------------- live badge ---------------- */
      .sf-live {
        display:inline-flex; align-items:center; gap:.42rem;
        background:rgba(255,90,95,.14); color:#FF8B8E;
        border:1px solid rgba(255,90,95,.42);
        padding:.18rem .66rem; border-radius:999px;
        font-size:.72rem; font-weight:700; letter-spacing:.05em;
      }
      .sf-live .dot {
        width:7px; height:7px; border-radius:50%; background:var(--sf-danger);
        animation:sfblink 1.1s ease-in-out infinite;
      }
      @keyframes sfblink { 0%,100%{opacity:1} 50%{opacity:.22} }

      /* ---------------- how-to steps ---------------- */
      .sf-steps { display:flex; flex-direction:column; gap:.5rem; margin:.2rem 0 .3rem; }
      .sf-step {
        display:flex; gap:.7rem; align-items:flex-start;
        background:var(--sf-surface); border:1px solid var(--sf-line);
        border-radius:12px; padding:.7rem .9rem;
      }
      .sf-step .n {
        flex:0 0 22px; height:22px; border-radius:50%;
        background:rgba(45,212,191,.14); color:var(--sf-primary);
        border:1px solid rgba(45,212,191,.38);
        display:flex; align-items:center; justify-content:center;
        font-size:.74rem; font-weight:800;
      }
      .sf-step .b { font-size:.87rem; color:var(--sf-ink); line-height:1.45; }
      .sf-step .b small { color:var(--sf-muted); }

      /* ---------------- sidebar ---------------- */
      section[data-testid="stSidebar"] {
        background:#080E1A; border-right:1px solid var(--sf-line);
      }
      section[data-testid="stSidebar"] * { color:#C6D4E6; }
      section[data-testid="stSidebar"] .sf-brand {
        font-size:1.08rem; font-weight:800; color:#fff; letter-spacing:-.01em;
      }
      section[data-testid="stSidebar"] .sf-brand-sub {
        font-size:.74rem; color:var(--sf-muted); margin:.08rem 0 .25rem;
      }
      section[data-testid="stSidebar"] hr { border-color:var(--sf-line); }

      section[data-testid="stSidebar"] div[role="radiogroup"] { gap:.22rem; }
      section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background:transparent; border:1px solid transparent;
        border-radius:10px; padding:.48rem .68rem; width:100%;
        transition:background .14s ease, border-color .14s ease;
      }
      section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background:rgba(255,255,255,.05);
      }
      section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background:rgba(45,212,191,.13); border-color:rgba(45,212,191,.40);
      }
      section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size:.89rem; font-weight:600;
      }

      .sf-side-card {
        background:#101A2B; border:1px solid var(--sf-line);
        border-radius:12px; padding:.65rem .8rem; margin-bottom:.5rem;
      }
      .sf-side-card .k { font-size:.66rem; color:var(--sf-muted); text-transform:uppercase;
                         letter-spacing:.08em; font-weight:700; }
      .sf-side-card .v { font-size:1.16rem; color:#fff; font-weight:800; }
      .sf-side-card .s { font-size:.7rem; color:var(--sf-muted); }
      .sf-ok { color:var(--sf-success) !important; font-weight:700; font-size:.8rem; }

      /* ---------------- streamlit chrome ---------------- */
      .stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--sf-line); }
      .stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.5rem 1rem; }
      div[data-testid="stMetricValue"] { font-size:1.45rem; font-weight:800; }
      .stProgress > div > div > div > div { background:var(--sf-primary); }
      div[data-testid="stExpander"] {
        border:1px solid var(--sf-line); border-radius:12px; background:var(--sf-surface);
      }
      div[data-testid="stFileUploaderDropzone"] {
        background:var(--sf-surface); border:1px dashed var(--sf-line);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the SafeFall AI model...")
def load_predictor() -> SafeFallPredictor:
    return SafeFallPredictor()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = config.RESULTS_DIR / "metrics_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return (
        json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        if config.METADATA_PATH.exists()
        else {}
    )


@st.cache_data(show_spinner=False)
def load_class_accuracy() -> dict:
    """Measured per-class precision on the held-out test videos.

    This is what makes a *live* accuracy read-out meaningful. Softmax confidence
    is the model's opinion of itself and is routinely over-confident; per-class
    precision is the measured share of times a given call turned out to be
    correct on videos the model never saw. Showing the figure for whichever
    class is currently predicted gives a number that updates live AND is backed
    by evidence.
    """
    path = config.RESULTS_DIR / "classification_report.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path, index_col=0)
    return {
        name: float(table.loc[name, "precision"])
        for name in config.CLASS_NAMES
        if name in table.index
    }


@st.cache_data(show_spinner=False)
def load_upper_body_accuracy() -> dict:
    """Measured accuracy of the upper-body detector (src/upper_body_train.py)."""
    path = config.RESULTS_DIR / "upper_body_eval.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


@st.cache_data(show_spinner=False, ttl=60)
def probe_camera() -> bool:
    """Cached so the dashboard does not reopen the camera on every rerun."""
    try:
        return camera_available(0)
    except Exception:
        return False


def init_session() -> None:
    defaults = {
        "total_activities": 0,
        "fall_count": 0,
        "normal_count": 0,
        "confidence_sum": 0.0,
        "activity_counts": {name: 0 for name in config.CLASS_NAMES},
        "event_log": [],
        "live_on": False,
        "live_state": None,
        "cctv_on": False,
        "cctv_state": None,
        "cctv_url": "",
        "cctv_tested": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def record(activity: str, confidence: float, source: str, detail: str = "") -> None:
    """Add one observation to the session-wide monitoring statistics."""
    st.session_state.total_activities += 1
    st.session_state.confidence_sum += float(confidence)
    st.session_state.activity_counts[activity] = (
        st.session_state.activity_counts.get(activity, 0) + 1
    )
    if activity == config.FALL_CLASS:
        st.session_state.fall_count += 1
    else:
        st.session_state.normal_count += 1
    st.session_state.event_log.append(
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Source": source,
            "Activity": activity,
            "Confidence": round(float(confidence), 4),
            "Status": "EMERGENCY" if activity == config.FALL_CLASS else "Safe",
            "Detail": detail,
        }
    )


# --------------------------------------------------------------------------- #
# Small view helpers
# --------------------------------------------------------------------------- #
def card(label: str, value: str, sub: str = "", icon: str = "", color: str = INK) -> str:
    ico = f'<span class="sf-ico">{icon}</span>' if icon else ""
    return (
        f'<div class="sf-card"><div class="sf-top">{ico}'
        f'<span class="sf-label">{label}</span></div>'
        f'<div class="sf-value" style="color:{color}">{value}</div>'
        f'<div class="sf-sub">{sub}</div></div>'
    )


def section(title: str) -> None:
    st.markdown(f'<div class="sf-section">{title}</div>', unsafe_allow_html=True)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def alert_tone() -> bytes:
    """A short two-tone siren generated on the fly - no audio assets needed."""
    rate = 22050
    duration = 0.28
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    tone = np.concatenate(
        [np.sin(2 * np.pi * 880 * t), np.sin(2 * np.pi * 660 * t)] * 2
    )
    envelope = np.minimum(1.0, np.linspace(0, 12, tone.size)) * np.minimum(
        1.0, np.linspace(12, 0, tone.size)
    )
    pcm = np.int16(np.clip(tone * envelope, -1, 1) * 22000)

    buffer = io.BytesIO()
    import wave

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def frame_html(frame_bgr: np.ndarray, quality: int = 60,
               max_width: int = 480) -> str:
    """Render a frame as an inline data URI.

    ``st.image`` writes every frame into Streamlit's media store and serves it
    over a separate HTTP request. At live frame rates the browser cannot finish
    fetching one frame before the next replaces it, so the feed shows as blank
    or flickers. Embedding the JPEG directly in the HTML removes the round-trip
    and the picture updates instantly.
    """
    import base64

    import cv2

    if max_width and frame_bgr.shape[1] > max_width:
        scale = max_width / frame_bgr.shape[1]
        frame_bgr = cv2.resize(
            frame_bgr, (max_width, int(frame_bgr.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", frame_bgr,
                              [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return (
        f'<img class="sf-feed" src="data:image/jpeg;base64,{encoded}" '
        f'alt="Live camera feed"/>'
    )


def framing_meter(score: float, ok: bool) -> str:
    """Report which detector is answering - not whether the user is misframed.

    Both views are supported: the whole body in shot unlocks the five-class
    CNN, and an upper-body view is handled by the trained upper-body detector.
    Colouring the second case as a warning implied the system had stopped
    working, when in fact it had simply switched engines, so it is shown in the
    ordinary accent colour with a label that says what is running.
    """
    pct = max(min(score, 1.0), 0.0) * 100
    if ok:
        colour = SUCCESS
        label = "Whole body in shot &middot; 5-class activity model"
    else:
        colour = PRIMARY
        label = "Upper-body view &middot; fall detection active"
    return (
        f'<div class="sf-frame-meter"><div class="t">'
        f'<span style="color:{colour}">VIEW &middot; {label}</span>'
        f'<span style="color:{colour}">{pct:.0f}%</span></div>'
        f'<div class="sf-frame-track"><div class="sf-frame-fill" '
        f'style="width:{max(pct, 2):.0f}%;background:{colour}"></div></div></div>'
    )


def live_panel_html(prediction, state) -> str:
    """Banner + metrics + probability bars as ONE block, for a single update."""
    if not prediction.pose_found:
        banner = ('<div class="sf-warn"><b>No person in view.</b><br>'
                  "Step into the camera's view, whole body visible.</div>")
        bars = ""
    elif not prediction.framing_ok:
        # Still answering - just with the upper-body detector rather than the CNN.
        banner = status_banner(prediction.activity, prediction.confidence,
                               prediction.message, "Upper-body detector",
                               state.alarm_active)
        bars = upper_body_bars(prediction.probabilities)
    else:
        banner = status_banner(prediction.activity, prediction.confidence,
                               prediction.message, "Live camera",
                               state.alarm_active)
        bars = probability_bars(prediction.probabilities)

    conf_colour = DANGER if state.alarm_active else INK
    alert_colour = DANGER if state.fall_events else INK

    # Live accuracy: the measured precision of whatever class is being predicted
    # right now. Falls back to a plain dash in upper-body mode, where the CNN's
    # per-class figures simply do not apply.
    class_accuracy = load_class_accuracy()
    if prediction.pose_found and prediction.framing_ok and \
            prediction.activity in class_accuracy:
        accuracy_value = f"{class_accuracy[prediction.activity]:.0%}"
        accuracy_sub = (f"measured precision for &lsquo;{prediction.activity}&rsquo; "
                        "on unseen test videos")
    elif prediction.pose_found and not prediction.framing_ok:
        upper = load_upper_body_accuracy()
        if upper:
            accuracy_value = f"{upper['test_accuracy']:.0%}"
            accuracy_sub = (f"upper-body detector &middot; measured on "
                            f"{upper['test_crops']} held-out crops &middot; "
                            f"fall recall {upper['test_fall_recall']:.0%}")
        else:
            accuracy_value = "n/a"
            accuracy_sub = "upper-body mode &mdash; geometric detector, not the CNN"
    else:
        accuracy_value = "--"
        accuracy_sub = "waiting for a person in view"

    cards = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;'
        'margin:.2rem 0 .6rem">'
        + card("Live accuracy", accuracy_value, accuracy_sub, "\U0001F3AF", PRIMARY)
        + card("Confidence", f"{prediction.confidence:.0%}",
               f"rolling avg {state.rolling_confidence:.0%}", "\U0001F4CA", conf_colour)
        + "</div>"
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;'
        'margin:0 0 .7rem">'
        + card("Fall alerts", f"{len(state.fall_events)}",
               f"{state.elapsed:.0f}s monitored", "\U0001F6A8", alert_colour)
        + card("Frames", f"{state.frames:,}",
               f"{state.fps:.1f} fps &middot; {state.usable_rate:.0%} full-body",
               "\U0001F39E\uFE0F")
        + "</div>"
        + f'<p class="sf-note" style="margin:.1rem 0 .6rem">Engine: '
          f'<b>{prediction.engine}</b></p>'
    )
    return banner + cards + bars


def upper_body_bars(probabilities: dict) -> str:
    """Fall / not-fall bars for the upper-body detector.

    The five-class bars are deliberately not reused here. This detector only
    ever answers one question, so rendering Sitting, Standing and Walking at
    0.0% would assert three things it has not measured.
    """
    fall = float(probabilities.get(config.FALL_CLASS, 0.0))
    rows = [
        ("\U0001F6A8 On the floor", fall, config.CLASS_STYLE[config.FALL_CLASS]["color"]),
        ("\U0001F9CD Upright", 1.0 - fall, config.CLASS_STYLE["Normal Activity"]["color"]),
    ]
    html = []
    for label, value, colour in rows:
        pct = value * 100
        html.append(
            f'<div class="sf-bar-row"><div class="sf-bar-top">'
            f'<span>{label}</span><span>{pct:.1f}%</span></div>'
            f'<div class="sf-bar-track"><div class="sf-bar-fill" '
            f'style="width:{max(pct, 0.6):.1f}%;background:{colour}"></div></div></div>'
        )
    return (f'<div class="sf-bars">{"".join(html)}</div>'
            '<p class="sf-note">Upper-body detector &mdash; it judges whether the '
            'person is on the floor, not which activity they are performing.</p>')


def probability_bars(probabilities: dict) -> str:
    """Lightweight HTML bars - fast enough to redraw on every live frame."""
    rows = []
    for name in config.CLASS_NAMES:
        pct = probabilities.get(name, 0.0) * 100
        colour = config.CLASS_STYLE[name]["color"]
        icon = config.CLASS_STYLE[name]["icon"]
        rows.append(
            f'<div class="sf-bar-row"><div class="sf-bar-top">'
            f'<span>{icon} {name}</span><span>{pct:.1f}%</span></div>'
            f'<div class="sf-bar-track"><div class="sf-bar-fill" '
            f'style="width:{max(pct, 0.6):.1f}%;background:{colour}"></div></div></div>'
        )
    return f'<div class="sf-bars">{"".join(rows)}</div>'


def status_banner(activity: str, confidence: float, message: str,
                  source: str, is_fall: bool) -> str:
    stamp = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    if is_fall:
        return f"""<div class="sf-alert">
          <h2>&#128680; EMERGENCY &mdash; FALL DETECTED</h2>
          <p><b>Confidence {confidence:.1%}</b> &nbsp;·&nbsp; {source} &nbsp;·&nbsp; {stamp}</p>
          <p>{message}</p>
          <p><b>Recommended action:</b> alert the on-duty caregiver immediately and
             check the resident for injury.</p>
        </div>"""
    icon = config.CLASS_STYLE.get(activity, {}).get("icon", "")
    return f"""<div class="sf-safe">
      <h2>{icon} STATUS: NORMAL &mdash; {activity.upper()}</h2>
      <p><b>Confidence {confidence:.1%}</b> &nbsp;·&nbsp; {source} &nbsp;·&nbsp; {stamp}</p>
      <p>{message}</p>
    </div>"""


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _base_layout(fig, height=330, title=None):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=14, t=46 if title else 16, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        # Always pass a real title dict - handing plotly ``None`` leaves the
        # title element in place and it renders as the literal text "undefined".
        title=dict(text=title or "", font=dict(size=14, color=INK)),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
    )
    return fig


def distribution_chart(counts: dict, title: str = "Activity distribution"):
    import plotly.graph_objects as go

    names = [n for n in config.CLASS_NAMES if counts.get(n, 0) > 0]
    if not names:
        return None
    fig = go.Figure(
        go.Pie(
            labels=names,
            values=[counts[n] for n in names],
            hole=0.62,
            marker=dict(colors=[config.CLASS_STYLE[n]["color"] for n in names],
                        line=dict(color="#0B1220", width=2)),
            textinfo="percent",
            hovertemplate="%{label}: %{value} detections (%{percent})<extra></extra>",
        )
    )
    fig = _base_layout(fig, 320, title)
    fig.update_layout(legend=dict(orientation="h", y=-0.08))
    return fig


def timeline_chart(timeline: pd.DataFrame, fall_events: list):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline["time"], y=timeline["fall_probability"] * 100,
        mode="lines", name="raw", line=dict(color="#5B6E8C", width=1.4),
    ))
    if "fall_probability_smoothed" in timeline.columns:
        fig.add_trace(go.Scatter(
            x=timeline["time"], y=timeline["fall_probability_smoothed"] * 100,
            mode="lines", name="smoothed",
            line=dict(color=DANGER, width=3), fill="tozeroy",
            fillcolor="rgba(220,38,38,.08)",
        ))
    fig.add_hline(y=config.FALL_PROB_THRESHOLD * 100, line_dash="dash",
                  line_color="#7C8FA8",
                  annotation_text=f"alert threshold ({config.FALL_PROB_THRESHOLD:.0%})",
                  annotation_position="top left")
    for event in fall_events:
        fig.add_vrect(x0=event["start_time"],
                      x1=max(event["end_time"], event["start_time"] + 0.1),
                      fillcolor=DANGER, opacity=0.12, line_width=0,
                      annotation_text="FALL", annotation_position="top left")
    fig = _base_layout(fig, 320)
    fig.update_layout(
        xaxis_title="Time (seconds)", yaxis_title="Fall probability (%)",
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor=GRID),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    return fig


def activity_ribbon(timeline: pd.DataFrame):
    import plotly.graph_objects as go

    detected = timeline[timeline["activity"] != "No person detected"]
    fig = go.Figure()
    for name in config.CLASS_NAMES:
        subset = detected[detected["activity"] == name]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["time"], y=[name] * len(subset), mode="markers",
            marker=dict(color=config.CLASS_STYLE[name]["color"], size=10, symbol="square"),
            name=name, hovertemplate="%{x:.1f}s · " + name + "<extra></extra>",
        ))
    fig = _base_layout(fig, 320)
    fig.update_layout(
        xaxis_title="Time (seconds)", showlegend=False,
        yaxis=dict(categoryorder="array", categoryarray=config.CLASS_NAMES[::-1]),
    )
    return fig


def live_chart(history: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["seconds"], y=history["fall_probability"] * 100,
        mode="lines", name="fall probability",
        line=dict(color=DANGER, width=2.6), fill="tozeroy",
        fillcolor="rgba(220,38,38,.09)",
    ))
    fig.add_trace(go.Scatter(
        x=history["seconds"], y=history["confidence"] * 100,
        mode="lines", name="confidence",
        line=dict(color=ACCENT, width=1.8, dash="dot"),
    ))
    fig.add_hline(y=config.FALL_PROB_THRESHOLD * 100, line_dash="dash",
                  line_color="#7C8FA8")
    fig = _base_layout(fig, 260)
    fig.update_layout(
        xaxis_title="Seconds since monitoring started", yaxis_title="%",
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor=GRID),
        legend=dict(orientation="h", y=1.2, x=0),
    )
    return fig


# --------------------------------------------------------------------------- #
# Shared result rendering
# --------------------------------------------------------------------------- #
def render_prediction(prediction, source: str, show_sound: bool) -> None:
    if not prediction.pose_found:
        st.markdown(
            '<div class="sf-warn"><b>No person detected.</b><br>'
            "The pose model could not find a human body. Use an image where the "
            "whole body is visible, reasonably lit and not heavily occluded.</div>",
            unsafe_allow_html=True,
        )
        return

    if not prediction.framing_ok:
        st.markdown(
            f'<div class="sf-warn"><b>Cannot classify &mdash; body not fully in shot.</b><br>'
            f"{prediction.framing_advice}<br><br>"
            "<small>Why this matters: the model reads posture from the hips, knees "
            "and ankles. When those are off-screen the pose estimator <i>guesses</i> "
            "where they would be, so any answer would be computed from coordinates "
            "that do not exist. Refusing to answer is the correct behaviour.</small>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        status_banner(prediction.activity, prediction.confidence,
                      prediction.message, source, prediction.is_fall),
        unsafe_allow_html=True,
    )
    if prediction.is_fall and show_sound:
        st.audio(alert_tone(), format="audio/wav", autoplay=True)

    geometry = prediction.geometry
    cols = st.columns(4)
    cells = [
        ("Predicted activity", prediction.activity, "5-class CNN output", "\U0001F9E0",
         DANGER if prediction.is_fall else INK),
        ("Confidence", f"{prediction.confidence:.1%}", "softmax probability", "\U0001F4CA", INK),
        ("Trunk angle", f"{geometry.get('torso_angle', 0):.0f}°",
         "0° upright · 90° horizontal", "\U0001F4D0", INK),
        ("Body aspect ratio", f"{geometry.get('bbox_aspect_ratio', 0):.2f}",
         "width / height · >1 = lying", "\U0001F4CF", INK),
    ]
    for col, (label, value, sub, icon, colour) in zip(cols, cells):
        col.markdown(card(label, value, sub, icon, colour), unsafe_allow_html=True)

    section("Class probabilities")
    st.markdown(probability_bars(prediction.probabilities), unsafe_allow_html=True)


def sample_files(suffixes: tuple) -> dict:
    """Bundled demo media, so the dashboard can be shown without any upload."""
    if not config.SAMPLES_DIR.exists():
        return {}
    found = {}
    for path in sorted(config.SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        kind = "Fall scenario" if "_fall_" in path.name else "Normal activity"
        found[f"{kind} · {path.stem.split('__')[-1]}"] = path
    return found


def analyse_image(frame, source: str, predictor: SafeFallPredictor, show_sound: bool) -> None:
    with st.spinner("Running pose estimation and classification..."):
        # Same framing rule as the live feed: a close-up photo where the legs
        # are out of shot gets an explanation, not a confident guess computed
        # from joints MediaPipe had to invent.
        prediction = predictor.predict_image(frame, draw=True, static=True,
                                             enforce_framing=True)

    left, right = st.columns(2)
    left.image(to_rgb(frame), caption="Input image", use_container_width=True)
    right.image(to_rgb(prediction.annotated_image),
                caption="Pose estimation · 33 body landmarks", use_container_width=True)

    render_prediction(prediction, source, show_sound)
    if prediction.pose_found:
        record(prediction.activity, prediction.confidence, source)
        with st.expander("Posture measurements behind this decision"):
            st.dataframe(
                pd.DataFrame({
                    "Feature": list(prediction.geometry.keys()),
                    "Value": [round(v, 3) for v in prediction.geometry.values()],
                }),
                use_container_width=True, hide_index=True, height=380,
            )


# --------------------------------------------------------------------------- #
# PAGE - Live camera
# --------------------------------------------------------------------------- #
def page_live(predictor: SafeFallPredictor, show_sound: bool) -> None:
    st.markdown(
        '<div class="sf-section">Laptop webcam monitoring</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sf-steps">'
        '<div class="sf-step"><div class="n">1</div><div class="b">'
        'Close any other app using the webcam (Teams, Zoom, Camera), then press '
        '<b>Start monitoring</b>.</div></div>'
        '<div class="sf-step"><div class="n">2</div><div class="b">'
        '<b>Stand back 2&ndash;3 metres so your whole body is in shot</b>, head to feet.'
        '<br><small>This matters more than anything else. The model reads posture from '
        'your hips, knees and ankles &mdash; it was trained on wall-mounted camera '
        'footage, not a close-up webcam. Watch the FRAMING bar turn green.</small>'
        '</div></div>'
        '<div class="sf-step"><div class="n">3</div><div class="b">'
        'Try walking, standing still, then sitting. Finally lie down on the floor &mdash; '
        'the alert fires once a fall holds for a few frames.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Skip the probe while streaming: it would open the device a second time
    # and force another warm-up cycle on the capture the loop is using.
    has_camera = True if st.session_state.live_on else probe_camera()

    if not has_camera:
        st.markdown(
            '<div class="sf-info"><b>No camera is attached to the machine running this app.</b><br>'
            "Continuous streaming reads the webcam on the server side, so it works when "
            "the dashboard runs on your own computer. On a cloud host there is no camera "
            "to open. Snapshot capture below still works anywhere, because it uses "
            "<i>your browser's</i> camera.</div>",
            unsafe_allow_html=True,
        )
        _snapshot_mode(predictor, show_sound)
        return

    controls = st.columns([1.1, 1, 1, 1.3])
    with controls[0]:
        if not st.session_state.live_on:
            if st.button("▶  Start monitoring", type="primary", use_container_width=True):
                st.session_state.live_on = True
                st.session_state.live_state = LiveMonitorState()
                st.rerun()
        else:
            if st.button("■  Stop monitoring", use_container_width=True):
                st.session_state.live_on = False
                st.rerun()
    with controls[1]:
        camera_index = st.number_input("Camera", 0, 5, 0, disabled=st.session_state.live_on)
    with controls[2]:
        mirror = st.toggle("Mirror view", value=True)
    with controls[3]:
        quality = st.select_slider(
            "Pose model", options=["Smooth", "Balanced", "Accurate"], value="Balanced",
            help="Smooth uses MediaPipe's lite pose network (~17 ms/frame); "
                 "Balanced and Accurate use the full one (~22 ms/frame). Every "
                 "frame is analysed either way.",
        )

    # Frame skipping is gone: the capture thread keeps only the newest frame, so
    # the loop can never fall behind and there is nothing to gain by dropping
    # frames. The slider now selects the pose model instead - the lite network
    # costs ~17 ms/frame against ~22 ms for the full one.
    skip, complexity = {"Smooth": (1, 0), "Balanced": (1, 1), "Accurate": (1, 1)}[quality]

    if not st.session_state.live_on:
        state = st.session_state.live_state
        if state is not None and state.frames:
            _live_summary(state)
        else:
            st.markdown(
                '<div class="sf-info">Press <b>Start monitoring</b> to begin. '
                "Stand back so your whole body is in frame, then try walking, standing "
                "still, sitting down, and finally lying down on the floor to trigger the "
                "emergency alert.</div>",
                unsafe_allow_html=True,
            )
        return

    _run_live_loop(predictor, int(camera_index), mirror, skip, show_sound,
                   state_key="live_state", flag_key="live_on",
                   source_label="Laptop webcam", complexity=complexity)


def _run_live_loop(predictor, source, mirror: bool, skip: int, show_sound: bool,
                   state_key: str = "live_state", flag_key: str = "live_on",
                   source_label: str = "Live camera", complexity: int = 1) -> None:
    """Drive the monitor from any OpenCV source - webcam index or stream URL."""
    import cv2

    state: LiveMonitorState = st.session_state.get(state_key) or LiveMonitorState()
    st.session_state[state_key] = state

    raw_cap = open_source(source)
    cap = FreshestFrame(raw_cap) if raw_cap is not None else None
    if cap is None:
        st.session_state[flag_key] = False
        if isinstance(source, int) or str(source).isdigit():
            st.error(
                f"Camera {source} could not be opened. It may be in use by another "
                "application (Teams, Zoom, the Camera app). Close that and try again."
            )
        else:
            st.error(
                f"Could not read {mask_credentials(str(source))}. Check the URL, "
                "the network, and the camera credentials."
            )
        return

    st.markdown(
        '<span class="sf-live"><span class="dot"></span>LIVE</span>',
        unsafe_allow_html=True,
    )

    feed_col, panel_col = st.columns([1.35, 1])
    frame_slot = feed_col.empty()
    panel_slot = panel_col.empty()
    chart_slot = st.empty()
    footer_slot = st.empty()

    predictor.reset_video_tracker(complexity=complexity)
    READ_STALL_TIMEOUT = 12.0        # seconds of no frames before giving up
    first_failure_at = None
    prev_head = None                 # (head_y, timestamp) for the head-drop cue
    pending_head_drop = 0.0
    raw_index = 0
    last_prediction = None
    alarm_sounded = False

    try:
        while st.session_state.get(flag_key) and state.elapsed < config.LIVE_MAX_SECONDS:
            ok, frame = cap.read()
            if not ok:
                # Dropped frames are normal: webcams stall while auto-exposure
                # adjusts and network cameras lose packets. Give the device a
                # generous window to recover, measured in seconds rather than in
                # frames, and only end the session if it really has gone away.
                if first_failure_at is None:
                    first_failure_at = time.time()
                stalled_for = time.time() - first_failure_at
                if stalled_for >= READ_STALL_TIMEOUT:
                    footer_slot.warning(
                        f"No frames for {stalled_for:.0f}s, so monitoring stopped. "
                        "This usually means another program or browser tab has "
                        "taken the camera - close it, then press Start again."
                    )
                    break
                footer_slot.info(f"Waiting for the camera... ({stalled_for:.0f}s)")
                time.sleep(0.08)
                continue
            if first_failure_at is not None:
                first_failure_at = None
                footer_slot.empty()

            if mirror:
                frame = cv2.flip(frame, 1)

            raw_index += 1

            head_drop = 0.0
            now_t = time.time()
            if prev_head is not None:
                gap = max(now_t - prev_head[1], 1e-3)
                head_drop = 0.0     # filled in below once we have the new head_y
            prediction = predictor.predict_image(
                frame, draw=True, static=False, enforce_framing=True,
                head_drop_per_sec=pending_head_drop,
            )
            head_y_now = prediction.geometry.get("head_y")
            if head_y_now is not None and prev_head is not None:
                gap = max(now_t - prev_head[1], 1e-3)
                pending_head_drop = max((head_y_now - prev_head[0]) / gap, 0.0)
            if head_y_now is not None:
                prev_head = (head_y_now, now_t)
            last_prediction = prediction
            just_fired = state.update(prediction)

            frame_slot.markdown(
                frame_html(prediction.annotated_image
                           if prediction.annotated_image is not None else frame)
                + framing_meter(prediction.framing_score, prediction.framing_ok),
                unsafe_allow_html=True,
            )

            # The picture needs every frame; the read-outs do not. Refreshing the
            # panel a few times a second instead of ~20 keeps the numbers legible
            # and removes most of the remaining DOM churn.
            if state.frames % config.LIVE_PANEL_EVERY == 0 or just_fired:
                panel_slot.markdown(live_panel_html(prediction, state),
                                    unsafe_allow_html=True)

            if state.frames % config.LIVE_CHART_EVERY == 0 and state.frames > 8:
                chart_slot.plotly_chart(live_chart(state.history_frame()),
                                        use_container_width=True)

            if just_fired and show_sound and not alarm_sounded:
                alarm_sounded = True
                footer_slot.error(
                    f"FALL CONFIRMED at {state.elapsed:.0f}s — "
                    "a caregiver would be paged now."
                )
            if not state.alarm_active:
                alarm_sounded = False

    except Exception as exc:  # noqa: BLE001
        # Anything unexpected ends this session cleanly rather than leaving the
        # page half-rendered with the camera still held open.
        st.session_state[flag_key] = False
        footer_slot.error(
            f"Live monitoring stopped after an unexpected error: {exc} - "
            "the camera has been released. Press Start to try again."
        )
    finally:
        try:
            cap.release()
        except Exception:
            pass

    if state.elapsed >= config.LIVE_MAX_SECONDS:
        st.session_state[flag_key] = False
        st.info(f"Monitoring stopped automatically after "
                f"{config.LIVE_MAX_SECONDS // 60} minutes.")

    if (last_prediction is not None and last_prediction.pose_found
            and last_prediction.framing_ok):
        record(last_prediction.activity, last_prediction.confidence, source_label)


def _live_summary(state: LiveMonitorState) -> None:
    section("Last live session")
    cols = st.columns(4)
    cols[0].markdown(card("Frames analysed", f"{state.frames:,}",
                          f"{state.elapsed:.0f} seconds", "\U0001F39E️"),
                     unsafe_allow_html=True)
    cols[1].markdown(card("Fall alerts", f"{len(state.fall_events)}",
                          "confirmed over consecutive frames", "\U0001F6A8",
                          DANGER if state.fall_events else INK),
                     unsafe_allow_html=True)
    cols[2].markdown(card("Average confidence", f"{state.rolling_confidence:.0%}",
                          "recent-window mean", "\U0001F3AF"), unsafe_allow_html=True)
    cols[3].markdown(card("Dominant activity", state.dominant_activity,
                          f"{state.detection_rate:.0%} of frames had a person",
                          "\U0001F464"), unsafe_allow_html=True)

    left, right = st.columns([1, 1.3])
    fig = distribution_chart(state.activity_counts, "Activity distribution")
    if fig:
        left.plotly_chart(fig, use_container_width=True)
    history = state.history_frame()
    if len(history) > 3:
        right.plotly_chart(live_chart(history), use_container_width=True)

    if state.fall_events:
        section("Fall alerts raised")
        st.dataframe(pd.DataFrame(state.fall_events).rename(columns={
            "at_seconds": "At (s)", "confidence": "Confidence", "frame": "Frame"}),
            use_container_width=True, hide_index=True)


def _snapshot_mode(predictor: SafeFallPredictor, show_sound: bool) -> None:
    section("Snapshot capture")
    st.caption(
        "Takes a single photo with your device camera and runs the full pipeline on it."
    )
    snapshot = st.camera_input("Capture a frame", key="camera_input")
    if snapshot is not None:
        import cv2

        data = np.frombuffer(snapshot.getvalue(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("That snapshot could not be decoded.")
            return
        with st.spinner("Analysing snapshot..."):
            prediction = predictor.predict_image(frame, draw=True, static=True,
                                                 enforce_framing=True)
        if prediction.annotated_image is not None:
            st.image(to_rgb(prediction.annotated_image),
                     caption="Pose estimation on the snapshot", use_container_width=True)
        render_prediction(prediction, "Camera snapshot", show_sound)
        if prediction.pose_found:
            record(prediction.activity, prediction.confidence, "Camera snapshot")




# --------------------------------------------------------------------------- #
# PAGE - Real-time camera over WebRTC (works on the deployed app)
# --------------------------------------------------------------------------- #
def _page_webrtc_camera(predictor: SafeFallPredictor, show_sound: bool) -> None:
    """Peer-to-peer media stream. Higher frame rate, but needs a relay."""
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        from src.webrtc_monitor import (
            TURN_ENV_VARS,
            LiveStats,
            make_frame_callback,
            turn_provider,
        )
    except Exception as exc:  # noqa: BLE001
        st.markdown(
            f'<div class="sf-warn"><b>Real-time streaming is unavailable.</b><br>'
            f'<code>{type(exc).__name__}: {exc}</code><br><br>'
            "An <code>ImportError</code> naming a <code>lib*.so</code> file means the "
            "host is missing a system library that OpenCV needs &mdash; add it to "
            "<code>packages.txt</code> and reboot. Everything else on this dashboard "
            "still works: use <b>Upload &amp; Analyse</b> or the snapshot capture."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="sf-info">This page streams <b>your own camera</b> from the '
        'browser to the app, so it gives real-time monitoring even on the '
        'deployed link &mdash; unlike the Laptop Webcam page, which can only '
        'reach a camera attached to the machine running the app.<br>'
        'Your browser will ask for camera permission. Video is processed and '
        'discarded frame by frame; nothing is recorded or stored.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sf-steps">'
        '<div class="sf-step"><div class="n">1</div><div class="b">'
        'Press <b>START</b> and allow camera access.</div></div>'
        '<div class="sf-step"><div class="n">2</div><div class="b">'
        '<b>Sit or stand wherever suits you.</b> Head and shoulders is enough '
        'to detect a fall.'
        '<br><small>The bar along the bottom of the video shows how much of you '
        'is in shot; with all of it the five-class activity model runs too.'
        '</small></div></div>'
        '<div class="sf-step"><div class="n">3</div><div class="b">'
        'Walk, stand, sit, then lie down &mdash; the banner turns red once a fall '
        'holds for several frames.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # streamlit-webrtc reads these with os.getenv. Streamlit Cloud usually
    # mirrors secrets into the environment already, but that is not guaranteed,
    # so copy them across explicitly before the streamer is built.
    for _name in TURN_ENV_VARS:
        if not os.environ.get(_name):
            try:
                _value = st.secrets[_name]
            except Exception:  # noqa: BLE001 - no secrets file is normal
                _value = None
            if _value:
                os.environ[_name] = str(_value)

    relay = turn_provider()
    if relay == "none":
        st.markdown(
            '<div class="sf-warn"><b>No video relay is configured, so this page '
            'may not connect on the deployed link.</b><br>'
            'A relay is what lets the browser and the app exchange video when '
            'they sit on different networks. Without one the stream stalls on '
            '&ldquo;Connection is taking longer than expected&rdquo;. Running '
            'the app on your own machine does not need it, because the browser '
            'and the app are then the same computer.<br><br>'
            'To switch one on, add a free <b>Hugging Face</b> access token '
            'under <b>Manage app &rarr; Settings &rarr; Secrets</b>:<br>'
            '<code>HF_TOKEN = "hf_your_token_here"</code><br>'
            'Save, reboot, and this page connects. No code changes needed.'
            '<br><br>Until then, <b>Upload &amp; Analyse</b> and the '
            '<b>Snapshot</b> tab work here exactly as they do locally.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="sf-info">Video relay active via <b>{relay}</b>. '
            'The relay forwards encrypted media only and cannot see your '
            'video.</div>',
            unsafe_allow_html=True,
        )

    controls = st.columns([1, 1, 2])
    with controls[0]:
        mirror = st.toggle("Mirror view", value=True, key="rtc_mirror")
    with controls[1]:
        quality = st.select_slider("Processing", options=["Smooth", "Balanced", "Accurate"],
                                   value="Balanced", key="rtc_quality")
    analyse_every = {"Smooth": 3, "Balanced": 2, "Accurate": 1}[quality]

    if "rtc_stats" not in st.session_state:
        st.session_state.rtc_stats = LiveStats()
    stats: LiveStats = st.session_state.rtc_stats

    feed_col, panel_col = st.columns([1.35, 1])

    with feed_col:
        try:
            ctx = webrtc_streamer(
                key="safefall-live",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                    "audio": False,
                },
                video_frame_callback=make_frame_callback(
                    predictor, stats, analyse_every=analyse_every, mirror=mirror
                ),
                async_processing=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.markdown(
                f'<div class="sf-warn"><b>Could not start the video stream.</b><br>'
                f'<code>{type(exc).__name__}: {exc}</code><br><br>'
                "The rest of the dashboard is unaffected &mdash; use "
                "<b>Upload &amp; Analyse</b> or the snapshot capture.</div>",
                unsafe_allow_html=True,
            )
            return

    with panel_col:
        placeholder = st.empty()

    if not ctx.state.playing:
        with panel_col:
            st.markdown(
                '<div class="sf-info">Press <b>START</b> under the video to begin.</div>',
                unsafe_allow_html=True)
        snap = stats.snapshot()
        if snap["frames"]:
            _render_rtc_summary(snap)
        return

    # While the stream runs, poll the shared stats and repaint the panel. The
    # frame callback lives on a WebRTC worker thread and must never touch
    # st.session_state, so this is the only safe direction for the data to flow.
    import time as _time

    for _ in range(240):                       # ~2 minutes per script run
        if not ctx.state.playing:
            break
        snap = stats.snapshot()
        placeholder.markdown(_rtc_panel_html(snap), unsafe_allow_html=True)
        _time.sleep(0.5)
    st.rerun()


def _rtc_panel_html(snap: dict) -> str:
    """Side panel for the real-time page, rendered as a single block."""
    if snap["alarm_active"]:
        banner = (f'<div class="sf-alert"><h2>&#128680; EMERGENCY &mdash; FALL DETECTED</h2>'
                  f'<p>Confirmed over {config.LIVE_CONFIRM_FRAMES} consecutive frames. '
                  "Dispatch a caregiver and check the resident for injury.</p></div>")
    elif snap["activity"] == "No person in view":
        banner = ('<div class="sf-warn"><b>No person in view.</b><br>'
                  "Step into the camera's view.</div>")
    elif not snap["framing_ok"]:
        icon = config.CLASS_STYLE.get(snap["activity"], {}).get("icon", "")
        banner = (f'<div class="sf-safe"><h2>{icon} {snap["activity"].upper()}</h2>'
                  f'<p>Confidence {snap["confidence"]:.0%} &middot; upper-body '
                  f'detector</p></div>')
    else:
        icon = config.CLASS_STYLE.get(snap["activity"], {}).get("icon", "")
        banner = (f'<div class="sf-safe"><h2>{icon} {snap["activity"].upper()}</h2>'
                  f'<p>Confidence {snap["confidence"]:.0%} &middot; status normal</p></div>')

    accuracy = load_class_accuracy()
    if snap["framing_ok"] and snap["activity"] in accuracy:
        acc_value = f"{accuracy[snap['activity']]:.0%}"
        acc_sub = f"measured precision for &lsquo;{snap['activity']}&rsquo;"
    elif snap["activity"] != "No person in view" and not snap["framing_ok"]:
        upper = load_upper_body_accuracy()
        acc_value = f"{upper['test_accuracy']:.0%}" if upper else "n/a"
        acc_sub = "upper-body detector"
    else:
        acc_value, acc_sub = "--", "waiting for a person"

    cards = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:.2rem 0 .6rem">'
        + card("Live accuracy", acc_value, acc_sub, "\U0001F3AF", PRIMARY)
        + card("Confidence", f"{snap['confidence']:.0%}",
               f"rolling avg {snap['rolling_confidence']:.0%}", "\U0001F4CA",
               DANGER if snap["alarm_active"] else INK)
        + "</div>"
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:0 0 .6rem">'
        + card("Fall alerts", f"{len(snap['fall_events'])}",
               f"{snap['elapsed']:.0f}s monitored", "\U0001F6A8",
               DANGER if snap["fall_events"] else INK)
        + card("Frames", f"{snap['frames']:,}",
               f"{snap['fps']:.1f} fps analysed", "\U0001F39E\uFE0F")
        + "</div>"
        + f'<p class="sf-note">Engine: <b>{snap["engine"] or "-"}</b></p>'
    )
    return banner + cards


def _render_rtc_summary(snap: dict) -> None:
    section("Last real-time session")
    cols = st.columns(3)
    cols[0].markdown(card("Frames analysed", f"{snap['frames']:,}",
                          f"{snap['elapsed']:.0f} seconds", "\U0001F39E\uFE0F"),
                     unsafe_allow_html=True)
    cols[1].markdown(card("Fall alerts", f"{len(snap['fall_events'])}",
                          "confirmed events", "\U0001F6A8",
                          DANGER if snap["fall_events"] else INK), unsafe_allow_html=True)
    cols[2].markdown(card("Full-body frames", f"{snap['frames_with_person'] - snap['frames_out_of_frame']:,}",
                          f"of {snap['frames_with_person']:,} with a person",
                          "\U0001F464"), unsafe_allow_html=True)
    fig = distribution_chart(snap["activity_counts"], "Activity distribution")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# PAGE - Real-time camera. Two transports, same analysis.
# --------------------------------------------------------------------------- #
def page_realtime(predictor: SafeFallPredictor, show_sound: bool) -> None:
    st.markdown('<div class="sf-section">Real-time camera monitoring</div>',
                unsafe_allow_html=True)

    mode = st.radio(
        "How should the video reach the app?",
        ["Direct stream", "WebRTC"],
        horizontal=True,
        key="rt_transport",
        captions=[
            "Works on the deployed link with no setup. A few frames per second.",
            "Full frame rate, but needs a TURN relay to be configured.",
        ],
    )
    if mode == "Direct stream":
        _page_direct_camera(predictor, show_sound)
    else:
        _page_webrtc_camera(predictor, show_sound)


CAMERA_KEY = "safefall_browser_camera"


def _page_direct_camera(predictor: SafeFallPredictor, show_sound: bool) -> None:
    """Live monitoring without WebRTC, and therefore without a relay.

    The browser captures each frame and hands it to this script over the
    component channel - the same websocket the dashboard already uses. Every
    frame costs a rerun, so the rate is a few per second rather than 15-30,
    which the alarm absorbs because it confirms over consecutive frames.
    """
    try:
        from src.browser_camera import browser_camera, decode_frame
        from src.live_camera import LiveMonitorState
    except Exception as exc:  # noqa: BLE001
        st.markdown(
            f'<div class="sf-warn"><b>Direct streaming is unavailable.</b><br>'
            f'<code>{type(exc).__name__}: {exc}</code></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="sf-info">Your browser opens the camera and sends each frame '
        'to the app over the same connection the dashboard already uses, so this '
        'works on the deployed link with <b>no relay and no configuration</b>.<br>'
        'Frames are analysed and discarded as they arrive; nothing is recorded '
        'or stored.</div>',
        unsafe_allow_html=True,
    )

    if "direct_state" not in st.session_state:
        st.session_state.direct_state = LiveMonitorState()
        st.session_state.direct_seq = None
        st.session_state.direct_last = None

    controls = st.columns([1, 1, 1.5, 1])
    with controls[0]:
        running = st.toggle("Camera on", value=False, key="direct_running")
    with controls[1]:
        mirror = st.toggle("Mirror view", value=True, key="direct_mirror")
    with controls[2]:
        detail = st.select_slider("Detail", options=["Fast", "Balanced", "Sharp"],
                                  value="Balanced", key="direct_detail")
    with controls[3]:
        if st.button("Reset session", key="direct_reset", use_container_width=True):
            st.session_state.direct_state = LiveMonitorState()
            st.session_state.direct_seq = None
            st.session_state.direct_last = None
    width = {"Fast": 320, "Balanced": 480, "Sharp": 640}[detail]

    st.markdown(
        '<div class="sf-steps">'
        '<div class="sf-step"><div class="n">1</div><div class="b">'
        'Switch <b>Camera on</b> and allow access when the browser asks.</div></div>'
        '<div class="sf-step"><div class="n">2</div><div class="b">'
        '<b>Sit or stand wherever suits you.</b> Head and shoulders is enough '
        'to detect a fall.'
        '<br><small>With your whole body in shot the five-class activity '
        'model runs as well, and the meter says which one is answering.'
        '</small></div></div>'
        '<div class="sf-step"><div class="n">3</div><div class="b">'
        'Walk, stand, sit, then lie down &mdash; the banner turns red once a fall '
        'holds for several frames.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Read the browser's latest payload from session state and analyse it
    # BEFORE re-rendering the component, so the acknowledgement it receives is
    # the frame just processed. Rendering first would acknowledge the previous
    # frame, the handshake would never release, and the stream would crawl
    # along on its safety-net timer instead.
    state = st.session_state.direct_state
    payload = st.session_state.get(CAMERA_KEY)

    camera_error = None
    just_fired = False
    if isinstance(payload, dict):
        if payload.get("error"):
            camera_error = payload["error"]
        elif (payload.get("frame")
                and payload.get("seq") != st.session_state.direct_seq):
            st.session_state.direct_seq = payload["seq"]
            frame = decode_frame(payload["frame"])
            if frame is not None:
                prediction = predictor.predict_image(
                    frame, draw=True, static=False, enforce_framing=True
                )
                just_fired = state.update(prediction)
                st.session_state.direct_last = prediction

    feed, panel = st.columns([1.3, 1])
    with feed:
        browser_camera(
            running=running, width=width, mirror=mirror,
            ack=int(st.session_state.direct_seq or 0), key=CAMERA_KEY,
        )

    if camera_error:
        with panel:
            st.markdown(
                '<div class="sf-warn"><b>The browser could not open the camera.</b>'
                f'<br><code>{camera_error}</code><br><br>'
                'Check the camera permission for this site, and that no other '
                'application is holding the camera.</div>',
                unsafe_allow_html=True,
            )
        return

    prediction = st.session_state.direct_last

    with panel:
        if prediction is None:
            st.markdown(
                '<div class="sf-info">Switch the camera on to begin. The first '
                'reading appears within a second or two.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                framing_meter(prediction.framing_score, prediction.framing_ok),
                unsafe_allow_html=True,
            )
            st.markdown(live_panel_html(prediction, state), unsafe_allow_html=True)

    if prediction is not None and prediction.annotated_image is not None:
        with feed:
            st.markdown(frame_html(prediction.annotated_image, max_width=520),
                        unsafe_allow_html=True)
            st.markdown(
                '<p class="sf-note">Latest analysed frame, with the 33 pose '
                'landmarks drawn on. The picture above is your camera at full '
                'rate; this one updates as each frame comes back from the '
                'model.</p>',
                unsafe_allow_html=True,
            )

    if just_fired and show_sound:
        st.audio(alert_tone(), format="audio/wav", autoplay=True)

    if not running and state.frames:
        _render_direct_summary(state)


def _render_direct_summary(state) -> None:
    section("Last session")
    cols = st.columns(3)
    cols[0].markdown(
        card("Frames analysed", f"{state.frames:,}",
             f"{state.elapsed:.0f} seconds monitored", "\U0001F39E\uFE0F"),
        unsafe_allow_html=True)
    cols[1].markdown(
        card("Fall alerts", f"{len(state.fall_events)}", "confirmed events",
             "\U0001F6A8", DANGER if state.fall_events else INK),
        unsafe_allow_html=True)
    cols[2].markdown(
        card("Frames with a person", f"{state.frames_with_person:,}",
             f"of {state.frames:,} analysed", "\U0001F464"),
        unsafe_allow_html=True)
    fig = distribution_chart(state.activity_counts, "Activity distribution")
    if fig:
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE - CCTV / IP camera
# --------------------------------------------------------------------------- #
def page_cctv(predictor: SafeFallPredictor, show_sound: bool) -> None:
    st.markdown('<div class="sf-section">CCTV / IP camera monitoring</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sf-info">This is the deployment the model was actually built '
        'for. Le2i training footage comes from <b>fixed cameras viewing a whole '
        'room</b>, so a wall-mounted CCTV camera matches the training data far '
        'better than a close-up laptop webcam &mdash; expect noticeably better '
        'accuracy here.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sf-steps">'
        '<div class="sf-step"><div class="n">1</div><div class="b">'
        'Paste your camera\'s stream URL below and press <b>Test connection</b>.'
        '</div></div>'
        '<div class="sf-step"><div class="n">2</div><div class="b">'
        'Aim the camera so the <b>whole room and a person head-to-toe</b> are in shot.'
        '<br><small>The FRAMING bar must be green for the system to classify.</small>'
        '</div></div>'
        '<div class="sf-step"><div class="n">3</div><div class="b">'
        'Press <b>Start monitoring</b>. Alerts latch after several confirmed frames.'
        '</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Common URL formats", expanded=False):
        st.markdown(
            """
| Camera | URL format |
|---|---|
| Generic CCTV / NVR (RTSP) | `rtsp://user:password@192.168.1.50:554/stream1` |
| Hikvision | `rtsp://user:password@192.168.1.50:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:password@192.168.1.50:554/cam/realmonitor?channel=1&subtype=0` |
| Android "IP Webcam" app | `http://192.168.1.30:8080/video` |
| MJPEG camera | `http://192.168.1.50/mjpg/video.mjpg` |

Find your camera's IP in your router's device list or the camera's own app. The
phone route is the easiest to demo: install **IP Webcam** on an Android phone,
tap *Start server*, and it shows you the URL.
"""
        )
        st.markdown(
            '<p class="sf-note">Most RTSP URLs contain the camera password. '
            'It is used only to open the stream, is never written to disk or logged, '
            'and is masked wherever it appears on screen.</p>',
            unsafe_allow_html=True,
        )

    url = st.text_input(
        "Camera stream URL",
        value=st.session_state.cctv_url,
        placeholder="rtsp://user:password@192.168.1.50:554/stream1",
        disabled=st.session_state.cctv_on,
        help="RTSP, HTTP or MJPEG. A plain number (0, 1) also works for a local device.",
    )
    st.session_state.cctv_url = url

    controls = st.columns([1, 1.1, 1, 1.2])
    with controls[0]:
        if st.button("Test connection", use_container_width=True,
                     disabled=st.session_state.cctv_on or not url.strip()):
            with st.spinner("Connecting..."):
                ok, message, frame = test_stream(url)
            st.session_state.cctv_tested = (ok, message)
            if ok and frame is not None:
                st.session_state["cctv_preview"] = frame
    with controls[1]:
        if not st.session_state.cctv_on:
            if st.button("▶  Start monitoring", type="primary",
                         use_container_width=True, disabled=not url.strip()):
                st.session_state.cctv_on = True
                st.session_state.cctv_state = LiveMonitorState()
                st.rerun()
        else:
            if st.button("■  Stop monitoring", use_container_width=True):
                st.session_state.cctv_on = False
                st.rerun()
    with controls[2]:
        mirror = st.toggle("Mirror view", value=False, key="cctv_mirror")
    with controls[3]:
        quality = st.select_slider(
            "Pose model", options=["Smooth", "Balanced", "Accurate"],
            value="Balanced", key="cctv_quality",
            help="Smooth uses MediaPipe's lite pose network; the others use the "
                 "full one. Every frame is analysed either way.",
        )
    skip, complexity = {"Smooth": (1, 0), "Balanced": (1, 1), "Accurate": (1, 1)}[quality]

    tested = st.session_state.cctv_tested
    if tested is not None and not st.session_state.cctv_on:
        ok, message = tested
        css = "sf-safe" if ok else "sf-warn"
        title = "Connection OK" if ok else "Could not connect"
        st.markdown(f'<div class="{css}"><h2>{title}</h2><p>{message}</p></div>',
                    unsafe_allow_html=True)
        preview = st.session_state.get("cctv_preview")
        if ok and preview is not None:
            st.markdown(frame_html(preview), unsafe_allow_html=True)

    if not st.session_state.cctv_on:
        state = st.session_state.cctv_state
        if state is not None and state.frames:
            _live_summary(state)
        return

    _run_live_loop(predictor, url, mirror, skip, show_sound,
                   state_key="cctv_state", flag_key="cctv_on",
                   source_label="CCTV camera", complexity=complexity)

# --------------------------------------------------------------------------- #
# PAGE - Uploads
# --------------------------------------------------------------------------- #
def page_monitor(predictor: SafeFallPredictor, show_sound: bool) -> None:
    section("Analyse a photo or a recording")
    tab_image, tab_video, tab_snapshot = st.tabs(
        ["\U0001F5BC️  Image", "\U0001F3A5  Video", "\U0001F4F7  Snapshot"]
    )

    with tab_image:
        st.caption(
            "Upload a photo of the monitored room. The system finds the body, "
            "classifies the activity and raises an emergency alert if it detects a fall."
        )
        uploaded = st.file_uploader("Choose an image",
                                    type=["jpg", "jpeg", "png", "bmp", "webp"],
                                    key="image_upload")

        samples = sample_files((".jpg", ".jpeg", ".png"))
        chosen = None
        if samples:
            with st.expander("No image to hand? Try a bundled sample",
                             expanded=uploaded is None):
                choice = st.selectbox("Sample image", list(samples), key="sample_image")
                if st.button("Analyse sample image", key="run_sample_image",
                             type="primary"):
                    chosen = samples[choice]

        import cv2

        if uploaded is not None:
            data = np.frombuffer(uploaded.getvalue(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                st.error("That file could not be decoded as an image.")
            else:
                analyse_image(frame, f"Image upload: {uploaded.name}", predictor, show_sound)
        elif chosen is not None:
            frame = cv2.imread(str(chosen))
            if frame is None:
                st.error("The bundled sample could not be read.")
            else:
                analyse_image(frame, f"Sample image: {chosen.name}", predictor, show_sound)

    with tab_video:
        st.caption(
            "Upload a monitoring clip. Every sampled frame is classified, the fall "
            "probability is smoothed over time, and an alert is only confirmed when the "
            "signal persists — which is what removes single-frame false alarms."
        )
        uploaded = st.file_uploader("Choose a video",
                                    type=["mp4", "avi", "mov", "mkv", "webm"],
                                    key="video_upload")

        samples = sample_files((".mp4", ".avi"))
        chosen = None
        if samples:
            with st.expander("No clip to hand? Try a bundled sample",
                             expanded=uploaded is None):
                st.caption(
                    "These short clips come from held-out recordings the model never "
                    "trained on — two real falls and one ordinary activity clip."
                )
                choice = st.selectbox("Sample clip", list(samples), key="sample_video")
                if st.button("Analyse sample clip", key="run_sample_video",
                             type="primary"):
                    chosen = samples[choice]

        temp_path = source_name = None
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(uploaded.getvalue())
                temp_path = handle.name
            source_name = uploaded.name
        elif chosen is not None:
            temp_path, source_name = str(chosen), chosen.name

        if temp_path is not None:
            _analyse_video(predictor, temp_path, source_name,
                           uploaded is not None, show_sound)

    with tab_snapshot:
        _snapshot_mode(predictor, show_sound)


def _analyse_video(predictor, temp_path: str, source_name: str,
                   is_upload: bool, show_sound: bool) -> None:
    progress = st.progress(0.0, text="Analysing video...")
    try:
        analysis = predictor.analyse_video(
            temp_path,
            progress_callback=lambda p: progress.progress(
                min(p, 1.0), text=f"Analysing video... {p:.0%}"),
        )
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        st.error(f"Could not analyse this video: {exc}")
        return
    progress.empty()

    summary = analysis.summary
    if summary["emergency"]:
        first = analysis.fall_events[0]
        st.markdown(f"""<div class="sf-alert">
          <h2>&#128680; EMERGENCY &mdash; {len(analysis.fall_events)} FALL EVENT(S) DETECTED</h2>
          <p>First event at <b>{first['start_time']:.1f}s</b>, lasting
             <b>{first['duration']:.1f}s</b>, peak confidence
             <b>{first['peak_confidence']:.1%}</b>, severity <b>{first['severity']}</b>.</p>
          <p><b>Recommended action:</b> dispatch a caregiver to the monitored room
             and check the resident for injury.</p>
        </div>""", unsafe_allow_html=True)
        if show_sound:
            st.audio(alert_tone(), format="audio/wav", autoplay=True)
    else:
        st.markdown(f"""<div class="sf-safe">
          <h2>&#9989; STATUS: NORMAL &mdash; NO FALL DETECTED</h2>
          <p>{summary['total_activities_detected']} activity observations across
             {analysis.duration_seconds:.1f}s of footage. Dominant activity:
             <b>{summary['dominant_activity']}</b>.</p>
        </div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].markdown(card("Total activities", f"{summary['total_activities_detected']:,}",
                          f"{summary['total_frames_analysed']:,} frames sampled",
                          "\U0001F4CB"), unsafe_allow_html=True)
    cols[1].markdown(card("Fall frames", f"{summary['fall_frame_count']:,}",
                          f"{summary['fall_events']} confirmed event(s)", "\U0001F6A8",
                          DANGER if summary["fall_frame_count"] else INK),
                     unsafe_allow_html=True)
    cols[2].markdown(card("Normal activity", f"{summary['normal_activity_count']:,}",
                          "walking · sitting · standing", "\U0001F6B6", SUCCESS),
                     unsafe_allow_html=True)
    cols[3].markdown(card("Average confidence", f"{summary['average_confidence']:.1%}",
                          f"peak fall probability {summary['peak_fall_probability']:.0%}",
                          "\U0001F3AF"), unsafe_allow_html=True)

    timeline = pd.DataFrame(analysis.timeline)

    section("Fall probability over time")
    st.plotly_chart(timeline_chart(timeline, analysis.fall_events),
                    use_container_width=True)

    left, right = st.columns(2)
    fig = distribution_chart(summary["activity_counts"], "Activity distribution")
    if fig:
        left.plotly_chart(fig, use_container_width=True)
    with right:
        section("Detected activity per second")
        st.plotly_chart(activity_ribbon(timeline), use_container_width=True)

    if analysis.fall_events:
        section("Confirmed fall events")
        st.dataframe(
            pd.DataFrame(analysis.fall_events).rename(columns={
                "start_time": "Start (s)", "end_time": "End (s)",
                "duration": "Duration (s)", "peak_confidence": "Peak confidence",
                "peak_time": "Peak at (s)", "frames": "Frames",
                "rapid_descent": "Rapid descent", "severity": "Severity",
                "descent_velocity": "Descent velocity",
            }).drop(columns=["peak_frame"], errors="ignore"),
            use_container_width=True, hide_index=True,
        )

    if analysis.key_frames:
        section("Pose visualisation · key frames")
        for start in range(0, len(analysis.key_frames), 3):
            for col, item in zip(st.columns(3), analysis.key_frames[start:start + 3]):
                flag = "\U0001F6A8 " if item["is_fall"] else ""
                col.image(to_rgb(item["image"]),
                          caption=f"{flag}{item['time']:.1f}s · {item['activity']} "
                                  f"({item['confidence']:.0%})",
                          use_container_width=True)

    with st.expander("Frame-by-frame results"):
        st.dataframe(timeline, use_container_width=True, height=340)
    st.download_button("⬇️  Download incident report (CSV)",
                       timeline.to_csv(index=False).encode("utf-8"),
                       file_name=f"safefall_report_{Path(source_name).stem}.csv",
                       mime="text/csv")

    for name, count in summary["activity_counts"].items():
        st.session_state.activity_counts[name] = (
            st.session_state.activity_counts.get(name, 0) + count
        )
    st.session_state.total_activities += summary["total_activities_detected"]
    st.session_state.fall_count += summary["fall_frame_count"]
    st.session_state.normal_count += summary["normal_activity_count"]
    st.session_state.confidence_sum += (
        summary["average_confidence"] * summary["total_activities_detected"]
    )
    st.session_state.event_log.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Source": "Video upload" if is_upload else "Sample clip",
        "Activity": summary["dominant_activity"],
        "Confidence": round(summary["average_confidence"], 4),
        "Status": "EMERGENCY" if summary["emergency"] else "Safe",
        "Detail": f"{source_name} — {summary['fall_events']} fall event(s)",
    })
    if is_upload:
        Path(temp_path).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# PAGE - Analytics
# --------------------------------------------------------------------------- #
def page_analytics() -> None:
    section("Monitoring analytics · this session")

    total = st.session_state.total_activities
    falls = st.session_state.fall_count
    normal = st.session_state.normal_count
    avg_conf = st.session_state.confidence_sum / total if total else 0.0

    cols = st.columns(4)
    cols[0].markdown(card("Total activities detected", f"{total:,}",
                          "across every analysis this session", "\U0001F4CB"),
                     unsafe_allow_html=True)
    cols[1].markdown(card("Falls detected", f"{falls:,}", "emergency observations",
                          "\U0001F6A8", DANGER if falls else INK), unsafe_allow_html=True)
    cols[2].markdown(card("Normal activities", f"{normal:,}",
                          "walking · sitting · standing", "\U0001F6B6", SUCCESS),
                     unsafe_allow_html=True)
    cols[3].markdown(card("Average confidence", f"{avg_conf:.1%}",
                          "mean softmax confidence", "\U0001F3AF"), unsafe_allow_html=True)

    if total == 0:
        st.markdown(
            '<div class="sf-info">No activity recorded yet. Run the <b>Live Camera</b> '
            "or analyse something on the <b>Monitor</b> page and the analytics will "
            "populate here.</div>", unsafe_allow_html=True)
        return

    left, right = st.columns([1, 1.25])
    fig = distribution_chart(st.session_state.activity_counts, "Activity distribution")
    if fig:
        left.plotly_chart(fig, use_container_width=True)

    with right:
        import plotly.graph_objects as go

        counts = st.session_state.activity_counts
        names = [n for n in config.CLASS_NAMES if counts.get(n, 0) > 0]
        fig = go.Figure(go.Bar(
            x=names, y=[counts[n] for n in names],
            marker_color=[config.CLASS_STYLE[n]["color"] for n in names],
            text=[counts[n] for n in names], textposition="outside",
        ))
        fig = _base_layout(fig, 320, "Detections per activity class")
        st.plotly_chart(fig, use_container_width=True)

    safety = 100.0 * (1 - falls / total) if total else 100.0
    section("Resident safety index")
    st.progress(min(max(safety / 100, 0.0), 1.0),
                text=f"{safety:.1f}% of observations were safe, non-fall activity")

    section("Event log")
    log = pd.DataFrame(st.session_state.event_log)
    st.dataframe(log, use_container_width=True, hide_index=True, height=320)
    st.download_button("⬇️  Download event log (CSV)",
                       log.to_csv(index=False).encode("utf-8"),
                       file_name="safefall_event_log.csv", mime="text/csv")
    if st.button("Reset session statistics"):
        for key in ("total_activities", "fall_count", "normal_count",
                    "confidence_sum", "activity_counts", "event_log"):
            st.session_state.pop(key, None)
        st.rerun()


# --------------------------------------------------------------------------- #
# PAGE - Model performance
# --------------------------------------------------------------------------- #
def page_performance() -> None:
    metrics = load_metrics()
    metadata = load_metadata()

    section("Model performance on unseen test videos")
    if not metrics:
        st.warning("Run `python -m src.evaluate` to generate the evaluation artefacts.")
        return

    cols = st.columns(4)
    for col, (label, key, sub) in zip(cols, [
        ("Accuracy", "test_accuracy", "overall correct predictions"),
        ("Precision (macro)", "test_precision_macro", "averaged over the 5 classes"),
        ("Recall (macro)", "test_recall_macro", "averaged over the 5 classes"),
        ("F1-score (macro)", "test_f1_macro", "harmonic mean of the two"),
    ]):
        col.markdown(card(label, f"{metrics[key]:.2%}", sub, "\U0001F4CA"),
                     unsafe_allow_html=True)

    section("Fall class · the metrics that matter clinically")
    cols = st.columns(4)
    cols[0].markdown(card("Fall recall (sensitivity)", f"{metrics.get('fall_recall', 0):.2%}",
                          "share of real falls caught", "\U0001F6A8", DANGER),
                     unsafe_allow_html=True)
    cols[1].markdown(card("Fall precision", f"{metrics.get('fall_precision', 0):.2%}",
                          "share of alerts that were real", "\U0001F3AF"),
                     unsafe_allow_html=True)
    cols[2].markdown(card("Fall F1-score", f"{metrics.get('fall_f1', 0):.2%}",
                          "balance of the two", "⚖️"), unsafe_allow_html=True)
    cols[3].markdown(card("Fall ROC-AUC", f"{metrics.get('fall_roc_auc', 0):.3f}",
                          "ranking quality · 1.0 = perfect", "\U0001F4C8"),
                     unsafe_allow_html=True)

    st.markdown(
        f'<p class="sf-note">Evaluated on {metrics.get("test_frames", 0):,} frames from '
        f'{metrics.get("test_videos", 0)} videos never seen during training. The split is '
        "made at video level, so no frame of a test recording ever appeared in the "
        f"training data. At the deployed alert threshold of "
        f"{metrics.get('alert_threshold', 0):.0%}, fall recall rises to "
        f"<b>{metrics.get('fall_recall_at_alert_threshold', 0):.2%}</b>.</p>",
        unsafe_allow_html=True,
    )

    images = [
        ("confusion_matrix.png", "Confusion matrix"),
        ("per_class_metrics.png", "Precision · recall · F1 per activity"),
        ("accuracy_graph.png", "Accuracy per epoch"),
        ("loss_graph.png", "Loss per epoch"),
        ("fall_detection_curves.png", "ROC and precision-recall for the fall class"),
        ("threshold_analysis.png", "Choosing the emergency-alert threshold"),
        ("fall_sequence.png", "A real fall, frame by frame"),
        ("misclassification_examples.png", "Where the model gets it wrong"),
    ]
    available = [(config.RESULTS_DIR / f, t) for f, t in images
                 if (config.RESULTS_DIR / f).exists()]
    for i in range(0, len(available), 2):
        for col, (path, title) in zip(st.columns(2), available[i:i + 2]):
            with col:
                section(title)
                st.image(str(path), use_container_width=True)

    analysis = load_text(config.RESULTS_DIR / "confusion_matrix_analysis.txt")
    if analysis:
        with st.expander("Written analysis of the confusion matrix"):
            st.code(analysis, language="text")
    report = load_text(config.RESULTS_DIR / "classification_report.txt")
    if report:
        with st.expander("Full classification report"):
            st.code(report, language="text")
    if metadata:
        with st.expander("Training configuration"):
            st.json(metadata)

    if "unseen_scene_agreement" in metrics:
        section("Generalisation to completely unseen rooms")
        st.markdown(
            f'<p class="sf-note">The model was additionally run on '
            f'<b>{metrics.get("unseen_scene_frames", 0):,} frames from '
            f'{metrics.get("unseen_scene_videos", 0)} videos</b> recorded in two rooms that '
            "appear nowhere in training, validation or test (Lecture room and Office, with "
            "different lighting, furniture and camera angles). Agreement with the reference "
            f'labels there is <b>{metrics["unseen_scene_agreement"]:.2%}</b> — essentially '
            "identical to the test split, which is the clearest evidence that the model is "
            "not memorising a room.</p>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# PAGE - About
# --------------------------------------------------------------------------- #
def page_about() -> None:
    metadata = load_metadata()
    section("How SafeFall AI works")
    st.markdown("""
```
  Camera frame / uploaded media
            |
            v
  [1] MediaPipe Pose  ->  33 body landmarks (shoulders, elbows, wrists,
      (BlazePose)          hips, knees, ankles) with x, y, z, visibility
            |
            v
  [2] Feature engineering
      Branch A: hip-centred, torso-scaled skeleton tensor  (33 x 4)
      Branch B: 25 clinical posture features (trunk angle, knee angle,
                bounding-box aspect ratio, stance width, ...)
            |
            v
  [3] Two-branch CNN classifier
      1-D convolutions along the kinematic chain + dense fusion head
            |
            v
  [4] Activity + confidence
      Fall Detected | Walking | Sitting | Standing | Normal Activity
            |
            v
  [5] Emergency logic
      live/video: probability smoothing + consecutive-frame confirmation
                  + rapid-descent severity check
      image:      fall probability above the alert threshold
            |
            v
  [6] Caregiver dashboard: alert, pose overlay, analytics, incident log
```
""")

    section("Why these models")
    st.markdown("""
- **MediaPipe Pose (BlazePose)** gives 33 full-body landmarks in real time on a
  plain CPU. Working from the skeleton rather than raw pixels means the system is
  far less sensitive to clothing colour, skin tone, furniture and wallpaper — and
  it is inherently more privacy-preserving, which matters a great deal in a care
  home.
- **A convolutional classifier** is the right family because a fall is defined by
  a *local geometric pattern* along the body — trunk rotating toward horizontal,
  hips dropping, legs collapsing. 1-D convolutions along the kinematic chain share
  weights across body parts exactly the way 2-D convolutions share them across
  image patches.
- **The combination** is what makes this clinically relevant: pose estimation
  supplies a body-centric, lighting-robust representation, and the CNN turns it
  into an activity decision fast enough to run continuously on cheap hardware.
""")

    section("Known limitations")
    st.markdown("""
- **Single-frame ambiguity.** Walking and Standing look almost identical in one
  still image; the difference is motion. The live and video paths resolve this
  with temporal context, the image path cannot.
- **Occlusion.** If furniture hides the hips or legs, MediaPipe returns a partial
  skeleton and the system reports "no person detected" rather than guessing.
- **Camera angle and distance.** Landmarks are normalised by torso length, which
  handles distance well, but extreme overhead or very oblique angles remain harder.
- **Low light.** Pose detection rate drops in poorly lit rooms; an infrared or
  low-light camera would be the practical fix.
- **Weak labels.** Falls come from the Le2i ground-truth annotation files, but
  Walking / Sitting / Standing / Normal Activity are derived from transparent
  geometric rules because the dataset does not annotate them. The classifier
  therefore inherits the boundaries those rules draw, and the fall-class metrics
  are the properly supervised ones.
- **Dataset scope.** Le2i uses staged falls by younger actors in six rooms. Real
  elderly falls are slower, more varied and often partly occluded.
""")

    section("Planned improvements")
    st.markdown("""
- Add a temporal model (1-D CNN or LSTM over a window of frames) so Walking and
  Standing are separated by motion rather than posture alone.
- Collect and annotate genuine elderly-care footage, including slow "slump" falls
  and getting-out-of-bed events.
- Add low-light and infrared samples, and augment with brightness, blur and
  camera-angle jitter to harden the model.
- Reduce false alerts with a short "are you OK?" confirmation window before
  escalating, and with multi-camera agreement.
- Support real-time CCTV/RTSP streams and push notifications to caregiver phones.
- **Retraining cycle:** new footage is annotated, appended to the dataset, and the
  model is retrained and re-evaluated on a frozen test set every release. A drop
  in fall recall blocks the release.
""")

    if metadata:
        section("Model card")
        st.json(metadata)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
PAGES = {
    "\U0001F534  Live Camera (real-time)": "realtime",
    "\U0001F4BB  Laptop Webcam (local)": "live",
    "\U0001F4E1  CCTV / IP Camera": "cctv",
    "\U0001F4CA  Upload & Analyse": "monitor",
    "\U0001F4C8  Analytics": "analytics",
    "\U0001F52C  Model Performance": "performance",
    "ℹ️  About": "about",
}


def main() -> None:
    init_session()

    st.markdown("""
      <div class="sf-hero">
        <h1>&#128737;&#65039; SafeFall AI — Elderly Fall Detection &amp; Activity Monitoring</h1>
        <p>Pose-based deep-learning monitor that recognises falls, walking, sitting,
           standing and normal activity in real time, and raises an immediate
           emergency alert for caregivers.</p>
        <div class="sf-tags">
          <span class="sf-tag">MediaPipe Pose</span>
          <span class="sf-tag">Two-branch CNN</span>
          <span class="sf-tag">5 activity classes</span>
          <span class="sf-tag">Live camera monitoring</span>
          <span class="sf-tag">Real-time alerting</span>
        </div>
      </div>
    """, unsafe_allow_html=True)

    if not config.SCALER_PATH.exists() or not (
        config.NUMPY_WEIGHTS_PATH.exists() or config.KERAS_MODEL_PATH.exists()
    ):
        st.error(
            "Trained model files were not found. Run the pipeline first:\n\n"
            "```\npython -m src.build_dataset\npython -m src.labeling\n"
            "python -m src.split_dataset\npython -m src.train\n"
            "python -m src.export_model\n```"
        )
        st.stop()

    # OpenCV's native module is imported lazily by every page that analyses a
    # frame, so a host missing one of its shared libraries looks perfectly
    # healthy - model loaded, metrics rendered - right up until the first
    # analysis, and then dies with a redacted traceback on whichever page
    # happened to touch it first. Fail once, here, and say what is actually
    # wrong.
    try:
        import cv2  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"OpenCV could not be loaded: `{type(exc).__name__}: {exc}`\n\n"
            "The Python package is installed, but its native module needs system "
            "libraries this host does not have. Add the missing `lib*.so` to "
            "`packages.txt` and reboot the app - the full set OpenCV requires is "
            "tabulated in `docs/DEPLOYMENT_GUIDE.md`."
        )
        st.stop()

    predictor = load_predictor()
    metrics = load_metrics()

    with st.sidebar:
        st.markdown(
            '<div class="sf-brand">\U0001F3E5 SafeFall AI</div>'
            '<div class="sf-brand-sub">Elderly fall detection &amp; activity monitoring</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        page = PAGES[choice]
        st.divider()

        st.markdown(
            '<div class="sf-side-card"><div class="k">System status</div>'
            '<div class="sf-ok">● Model loaded and ready</div>'
            f'<div class="s">{predictor.engine_name}</div></div>',
            unsafe_allow_html=True,
        )
        if metrics:
            st.markdown(
                '<div class="sf-side-card"><div class="k">Test accuracy</div>'
                f'<div class="v">{metrics["test_accuracy"]:.1%}</div>'
                '<div class="s">on unseen videos</div></div>'
                '<div class="sf-side-card"><div class="k">Fall recall</div>'
                f'<div class="v">{metrics.get("fall_recall_at_alert_threshold", 0):.1%}</div>'
                '<div class="s">at the deployed alert threshold</div></div>',
                unsafe_allow_html=True,
            )
        st.divider()

        show_sound = st.toggle("Audible alarm on fall", value=True)
        st.caption(
            f"Alert threshold {config.FALL_PROB_THRESHOLD:.0%} · confirmed over "
            f"{config.LIVE_CONFIRM_FRAMES} consecutive live frames."
        )
        st.divider()
        st.markdown(
            '<div class="sf-side-card"><div class="k">This session</div>'
            f'<div class="v">{st.session_state.total_activities:,}</div>'
            '<div class="s">activities detected</div></div>'
            '<div class="sf-side-card"><div class="k">Falls detected</div>'
            f'<div class="v">{st.session_state.fall_count:,}</div>'
            '<div class="s">emergency observations</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("FA-2 · Machine Learning & Deep Learning")

    if page == "realtime":
        page_realtime(predictor, show_sound)
    elif page == "live":
        page_live(predictor, show_sound)
    elif page == "cctv":
        page_cctv(predictor, show_sound)
    elif page == "monitor":
        page_monitor(predictor, show_sound)
    elif page == "analytics":
        page_analytics()
    elif page == "performance":
        page_performance()
    else:
        page_about()


if __name__ == "__main__":
    main()
