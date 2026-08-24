"""
SafeFall AI - Central configuration.

Every path, class name, threshold and hyper-parameter used anywhere in the
project lives here so that the training pipeline and the deployed Streamlit
app can never drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
POSE_SAMPLES_DIR = RESULTS_DIR / "pose_samples"
PREDICTION_SAMPLES_DIR = RESULTS_DIR / "prediction_samples"

# Raw Le2i dataset carried forward from FA-1. Override with the
# SAFEFALL_RAW_DATASET environment variable if the folder ever moves.
RAW_DATASET_DIR = Path(
    os.environ.get(
        "SAFEFALL_RAW_DATASET",
        r"C:\Users\Admin\Desktop\AI\SafeFall_AI_FA1\Raw_Dataset",
    )
)

# Artefact file names
FEATURES_CSV = PROCESSED_DIR / "pose_features.csv"
UNSEEN_FEATURES_CSV = PROCESSED_DIR / "pose_features_unseen.csv"
VIDEO_INDEX_CSV = PROCESSED_DIR / "video_index.csv"

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

KERAS_MODEL_PATH = MODELS_DIR / "fall_detection_cnn.keras"
NUMPY_WEIGHTS_PATH = MODELS_DIR / "fall_detection_cnn_weights.npz"
SCALER_PATH = MODELS_DIR / "feature_scaler.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
HISTORY_CSV = RESULTS_DIR / "training_history.csv"

for _d in (
    PROCESSED_DIR,
    SPLITS_DIR,
    SAMPLES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    POSE_SAMPLES_DIR,
    PREDICTION_SAMPLES_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Activity classes (exactly the five outputs required by the FA-2 brief)
# --------------------------------------------------------------------------- #
CLASS_NAMES = [
    "Fall Detected",
    "Normal Activity",
    "Sitting",
    "Standing",
    "Walking",
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
FALL_CLASS = "Fall Detected"
FALL_INDEX = CLASS_TO_INDEX[FALL_CLASS]

# Colour + status metadata used by the dashboard
CLASS_STYLE = {
    "Fall Detected": {"color": "#E5383B", "icon": "\U0001F6A8", "status": "EMERGENCY"},
    "Normal Activity": {"color": "#4C9AFF", "icon": "\U0001F464", "status": "SAFE"},
    "Sitting": {"color": "#9B6DFF", "icon": "\U0001FA91", "status": "SAFE"},
    "Standing": {"color": "#F4A261", "icon": "\U0001F9CD", "status": "SAFE"},
    "Walking": {"color": "#2A9D8F", "icon": "\U0001F6B6", "status": "SAFE"},
}


# --------------------------------------------------------------------------- #
# Pose / feature extraction
# --------------------------------------------------------------------------- #
NUM_LANDMARKS = 33          # MediaPipe Pose (BlazePose) full-body landmark count
LANDMARK_CHANNELS = 4       # x, y, z, visibility
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5
POSE_MODEL_COMPLEXITY = 1   # 0=lite, 1=full, 2=heavy

# A frame is only used if the average landmark visibility clears this bar.
MIN_MEAN_VISIBILITY = 0.35

# Sample every Nth frame when building the training set (25 FPS source video).
FRAME_STRIDE = 3

# Names of the engineered geometric features (Branch B of the network).
GEOMETRIC_FEATURE_NAMES = [
    "torso_angle",             # angle of hip->shoulder vector from vertical (deg)
    "body_axis_angle",         # angle of ankle->shoulder vector from vertical (deg)
    "bbox_aspect_ratio",       # bounding-box width / height  (>1 => lying down)
    "bbox_height",             # normalised body height in the frame
    "bbox_width",              # normalised body width in the frame
    "hip_y",                   # vertical position of hips (0 = top, 1 = bottom)
    "shoulder_y",              # vertical position of shoulders
    "ankle_y",                 # vertical position of ankles
    "head_y",                  # vertical position of the nose
    "center_of_mass_y",        # mean y of all visible landmarks
    "hip_to_ankle_ratio",      # vertical hip->ankle distance / torso length
    "head_above_hip",          # normalised vertical gap head->hip (small when down)
    "left_knee_angle",         # hip-knee-ankle angle (deg), ~180 = straight
    "right_knee_angle",
    "mean_knee_angle",
    "left_hip_angle",          # shoulder-hip-knee angle (deg), small = sitting
    "right_hip_angle",
    "mean_hip_angle",
    "ankle_separation",        # |Lankle.x - Rankle.x| / shoulder width  (stride)
    "knee_separation",
    "shoulder_hip_width_ratio",
    "leg_asymmetry",           # vertical offset between ankles (gait cue)
    "arm_asymmetry",           # vertical offset between wrists (gait cue)
    "vertical_horizontal_ratio",
    "mean_visibility",
]
NUM_GEOMETRIC_FEATURES = len(GEOMETRIC_FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Rule-based (weak-supervision) labelling thresholds
# --------------------------------------------------------------------------- #
# Post-fall "still on the ground" detection, used to extend the ground-truth
# fall window for as long as the person remains down.
LYING_ASPECT_RATIO = 1.15
LYING_TORSO_ANGLE = 50.0
# Fraction of that person's own upright body height below which they are
# treated as being on the floor, even if the trunk is still vertical (the
# slumped-against-the-furniture case).
HEIGHT_COLLAPSE_RATIO = 0.62

# Standing - upright trunk, straight legs, feet close together.
STAND_MAX_TORSO_ANGLE = 28.0
STAND_MIN_KNEE_ANGLE = 145.0

# Walking - upright trunk, legs extended, feet apart in a stride stance.
WALK_MAX_TORSO_ANGLE = 28.0
WALK_MIN_KNEE_ANGLE = 132.0

# Standing and Walking share one stance-width boundary. Splitting them at a
# single value rather than leaving a dead band means borderline frames land in
# the *neighbouring upright class* instead of being dumped into the catch-all.
# Both are "safe" states that the dashboard treats identically, so a confusion
# there costs nothing clinically - whereas polluting the catch-all made it
# unlearnable.
STANCE_WIDTH_BOUNDARY = 0.65

# Normal Activity is defined positively, not as a residual: an upright person
# whose trunk is clearly inclined - bending, reaching, crouching, or in
# transition between sitting and standing. Clinically this is the posture band
# that carries the highest fall risk, so it earns its own class.
NORMAL_MIN_TORSO_ANGLE = 28.0
NORMAL_MAX_TORSO_ANGLE = 55.0

# Measured motion is NOT used to define the classes. It is used later, in the
# video pipeline, to refine an upright prediction once several frames are
# available (see inference.SafeFallPredictor._refine_with_motion).
MOTION_WALK_SPEED = 0.045        # normalised frame-widths per second
MOTION_STATIC_SPEED = 0.015

# Sitting
SIT_MAX_TORSO_ANGLE = 55.0
SIT_MAX_KNEE_ANGLE = 128.0
SIT_MAX_HIP_ANGLE = 125.0
SIT_MAX_ASPECT_RATIO = 1.10


# --------------------------------------------------------------------------- #
# Training hyper-parameters
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
# These defaults are the configuration selected on the *validation* split after
# comparing learning rates, batch sizes and augmentation strengths. The test
# split was only ever scored once, with these settings.
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 15
MAX_SAMPLES_PER_CLASS = 6000     # keeps the class distribution manageable
AUGMENT_COPIES = 4               # augmented skeletons generated per training frame
L2_REGULARISATION = 1e-4


# --------------------------------------------------------------------------- #
# Inference / alerting behaviour
# --------------------------------------------------------------------------- #
# Probability above which a single frame is treated as a fall candidate.
#
# Chosen on the VALIDATION split by the rule implemented in
# evaluate.choose_alert_threshold: among thresholds whose validation F1 for the
# fall class is within 0.5% of the maximum, take the one with the highest
# recall. Validation F1 is almost flat across 0.25-0.50, so that slack is spent
# on the error type that actually matters - a missed fall leaves a resident on
# the floor, a false alarm costs a caregiver five seconds.
# See results/threshold_analysis.png for the full curve.
FALL_PROB_THRESHOLD = 0.25
# How much validation F1 may be given up in exchange for recall.
THRESHOLD_F1_TOLERANCE = 0.005
# Number of consecutive smoothed frames required before the alarm latches.
# At 25 FPS with a stride of 2 this is about 0.4 s of sustained evidence -
# long enough to reject a single bad skeleton, short enough to alert fast.
FALL_CONFIRM_FRAMES = 5
# Width of the moving-average filter applied to per-frame probabilities.
SMOOTHING_WINDOW = 5
# Two alerts closer together than this are one incident, not two. A caregiver
# should be told "a fall happened", not handed four fragments of one event.
EVENT_MERGE_GAP_SECONDS = 2.0
# Downward hip velocity (normalised units per second) that counts as a
# rapid descent - the physical signature of a real fall.
RAPID_DESCENT_VELOCITY = 0.55
# Maximum number of frames the dashboard analyses from an uploaded video.
MAX_VIDEO_FRAMES = 900
VIDEO_FRAME_STRIDE = 2


# --------------------------------------------------------------------------- #
# Live camera monitoring (local machine only)
# --------------------------------------------------------------------------- #
# A continuous webcam feed is read server-side with OpenCV, so it works when the
# app runs on the same machine as the camera. On Streamlit Cloud the server has
# no webcam, and the dashboard falls back to single-snapshot mode automatically.
LIVE_MAX_SECONDS = 900          # hard stop, so a forgotten stream cannot run forever
LIVE_HISTORY_FRAMES = 240       # rolling window kept for the live charts
LIVE_ROLLING_WINDOW = 30        # frames averaged for the rolling confidence read-out
LIVE_CONFIRM_FRAMES = 4         # consecutive frames before the live alarm latches
LIVE_CHART_EVERY = 30           # redraw the live chart every N frames
LIVE_PANEL_EVERY = 4            # refresh the metric panel every N frames
LIVE_DEFAULT_WIDTH = 640
LIVE_DEFAULT_HEIGHT = 480


# --------------------------------------------------------------------------- #
# Inference-time accuracy options
# --------------------------------------------------------------------------- #
# Mirror test-time augmentation: average the prediction for a pose with the
# prediction for its mirror image. Every feature the model reads is an angle or
# a ratio, so a mirrored skeleton is genuinely the same posture - averaging the
# two cancels part of the network's left/right bias.
#
# Selected on the validation split (src/experiments.py): validation accuracy
# 0.8905 -> 0.8924 and fall recall 0.9352 -> 0.9501. On the held-out test split
# it is worth +0.7 points of accuracy and, far more importantly, +4.7 points of
# fall recall (0.8741 -> 0.9207). Cost is one extra forward pass through a
# 128k-parameter network.
USE_MIRROR_TTA = True
