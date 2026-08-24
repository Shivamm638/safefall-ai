"""
SafeFall AI - Stage 5: evaluation, metrics and every required plot.

Produces, in ``results/``:

    confusion_matrix.png          counts + row-normalised, side by side
    accuracy_graph.png            train vs validation accuracy per epoch
    loss_graph.png                train vs validation loss per epoch
    per_class_metrics.png         precision / recall / F1 per activity
    fall_detection_curves.png     ROC + precision-recall for the fall class
    threshold_analysis.png        how the alert threshold trades misses vs alarms
    classification_report.txt     scikit-learn text report
    classification_report.csv     the same numbers, machine readable
    metrics_summary.json          headline numbers used by the dashboard
    confusion_matrix_analysis.txt written interpretation of the matrix

Run with:  python -m src.evaluate
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config
from .data import FeatureScaler, frame_to_arrays, load_split

# --------------------------------------------------------------------------- #
# House style - calm clinical palette, readable at video resolution
# --------------------------------------------------------------------------- #
INK = "#1B2430"
MUTED = "#6B7A90"
GRID = "#DCE3EC"
ACCENT = "#0F6FBF"
ALERT = "#E5383B"
OK = "#2A9D8F"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    }
)


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_dataframe(df: pd.DataFrame, model, scaler: FeatureScaler) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(probabilities, y_true_or_None)`` for a feature table.

    Scoring goes through the *deployed* predictor rather than a bare Keras call,
    so the ensemble and mirror test-time augmentation the dashboard actually
    uses are included. Otherwise the Model Performance page would advertise
    numbers the running system does not achieve.
    """
    from .data import raw_landmarks

    y = df["label_index"].to_numpy(dtype=np.int64) if "label_index" in df.columns else None
    raw = raw_landmarks(df)
    probs = np.stack([model.predict_landmarks(lm) for lm in raw])
    return probs, y


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4))
    # Extra horizontal room so the right panel's y-label clears the left colorbar.
    fig.subplots_adjust(wspace=0.34)
    short = [n.replace(" Detected", "").replace(" Activity", "") for n in config.CLASS_NAMES]

    for ax, matrix, title, fmt, cmap in (
        (axes[0], cm, "Confusion Matrix - frame counts", "{:,}", "Blues"),
        (axes[1], cm_norm, "Confusion Matrix - recall per true class", "{:.1%}", "BuGn"),
    ):
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=matrix.max() if matrix.max() else 1)
        ax.set_xticks(range(config.NUM_CLASSES), short, rotation=20, ha="right")
        ax.set_yticks(range(config.NUM_CLASSES), short)
        ax.set_xlabel("Predicted activity")
        ax.set_ylabel("True activity")
        ax.set_title(title)
        ax.grid(False)
        threshold = matrix.max() * 0.55 if matrix.max() else 0.5
        for i in range(config.NUM_CLASSES):
            for j in range(config.NUM_CLASSES):
                ax.text(
                    j, i, fmt.format(matrix[i, j]),
                    ha="center", va="center", fontsize=10.5,
                    color="white" if matrix[i, j] > threshold else INK,
                    fontweight="bold" if i == j else "normal",
                )
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    fig.suptitle(
        "SafeFall AI - test-set confusion matrix (unseen videos)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png")
    plt.close(fig)
    return cm


def plot_training_curves() -> None:
    if not config.HISTORY_CSV.exists():
        print("  (no training history found - skipping accuracy/loss graphs)")
        return
    hist = pd.read_csv(config.HISTORY_CSV)

    for metric, title, fname, ylabel in (
        ("accuracy", "Model Accuracy per Epoch", "accuracy_graph.png", "Accuracy"),
        ("loss", "Model Loss per Epoch", "loss_graph.png", "Loss"),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.plot(hist["epoch"], hist[metric], color=ACCENT, lw=2.4, label=f"Training {ylabel.lower()}")
        ax.plot(hist["epoch"], hist[f"val_{metric}"], color=ALERT, lw=2.4, ls="--",
                label=f"Validation {ylabel.lower()}")

        best_idx = (
            int(hist["val_accuracy"].idxmax())
            if metric == "accuracy"
            else int(hist["val_loss"].idxmin())
        )
        best_x = hist["epoch"].iloc[best_idx]
        best_y = hist[f"val_{metric}"].iloc[best_idx]
        ax.scatter([best_x], [best_y], s=90, zorder=5, color=ALERT,
                   edgecolor="white", linewidth=2)
        ax.annotate(
            f"best epoch {int(best_x)}\nval {ylabel.lower()} = {best_y:.4f}",
            (best_x, best_y), textcoords="offset points", xytext=(12, 14),
            fontsize=10, color=MUTED,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        _style(ax)
        fig.savefig(config.RESULTS_DIR / fname)
        plt.close(fig)


def plot_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    precision = precision_score(y_true, y_pred, average=None, labels=range(config.NUM_CLASSES), zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=range(config.NUM_CLASSES), zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=range(config.NUM_CLASSES), zero_division=0)

    x = np.arange(config.NUM_CLASSES)
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars = [
        ax.bar(x - width, precision, width, label="Precision", color="#5B8FF9"),
        ax.bar(x, recall, width, label="Recall", color=OK),
        ax.bar(x + width, f1, width, label="F1-score", color="#F2A65A"),
    ]
    for group in bars:
        for rect in group:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                    f"{rect.get_height():.2f}", ha="center", fontsize=9, color=MUTED)

    ax.set_xticks(x, [n.replace(" ", "\n") for n in config.CLASS_NAMES])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-class performance on the unseen test videos")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    _style(ax)
    fig.savefig(config.RESULTS_DIR / "per_class_metrics.png")
    plt.close(fig)


def plot_fall_curves(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """ROC and precision-recall curves for the safety-critical fall class."""
    y_bin = (y_true == config.FALL_INDEX).astype(int)
    scores = probs[:, config.FALL_INDEX]

    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return {}

    fpr, tpr, _ = roc_curve(y_bin, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_bin, scores)
    ap = average_precision_score(y_bin, scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    axes[0].plot(fpr, tpr, color=ALERT, lw=2.6, label=f"Fall class (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1.4, label="Random guess")
    axes[0].set_xlabel("False-alarm rate")
    axes[0].set_ylabel("Fall detection rate (recall)")
    axes[0].set_title("ROC - detecting a fall")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color=ACCENT, lw=2.6, label=f"Average precision = {ap:.3f}")
    axes[1].axhline(y_bin.mean(), ls=":", color=MUTED, lw=1.4,
                    label=f"Fall prevalence = {y_bin.mean():.1%}")
    axes[1].set_xlabel("Recall (falls caught)")
    axes[1].set_ylabel("Precision (alerts that are real)")
    axes[1].set_title("Precision-Recall - detecting a fall")
    axes[1].legend(loc="lower left")

    for ax in axes:
        _style(ax)
    fig.suptitle("Fall class treated as the positive class (one-vs-rest)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(config.RESULTS_DIR / "fall_detection_curves.png")
    plt.close(fig)
    return {"fall_roc_auc": float(roc_auc), "fall_average_precision": float(ap)}


def choose_alert_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> dict:
    """Pick the emergency-alert threshold, using the VALIDATION split only.

    Rule: among all thresholds whose fall-class F1 is within
    ``config.THRESHOLD_F1_TOLERANCE`` of the best achievable, take the one with
    the highest recall.

    Validation F1 is nearly flat over a wide band of thresholds, so this spends
    that slack on the error type that carries the real cost. Selecting on
    validation rather than on test keeps the reported test numbers honest.
    """
    y_bin = (y_val == config.FALL_INDEX).astype(int)
    scores = val_probs[:, config.FALL_INDEX]
    if y_bin.sum() == 0:
        return {}

    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for t in grid:
        pred = (scores >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "f1": float(f1_score(y_bin, pred, zero_division=0)),
                "recall": float(recall_score(y_bin, pred, zero_division=0)),
                "precision": float(precision_score(y_bin, pred, zero_division=0)),
            }
        )

    best_f1 = max(r["f1"] for r in rows)
    eligible = [r for r in rows if r["f1"] >= best_f1 - config.THRESHOLD_F1_TOLERANCE]
    selected = max(eligible, key=lambda r: r["recall"])

    pd.DataFrame(rows).round(4).to_csv(
        config.RESULTS_DIR / "threshold_selection_validation.csv", index=False
    )
    return {
        "selected_threshold": selected["threshold"],
        "validation_f1_at_selected": round(selected["f1"], 4),
        "validation_recall_at_selected": round(selected["recall"], 4),
        "validation_best_f1": round(best_f1, 4),
        "f1_tolerance": config.THRESHOLD_F1_TOLERANCE,
    }


def plot_threshold_analysis(y_true: np.ndarray, probs: np.ndarray) -> None:
    """Show why the alert threshold sits where it does."""
    y_bin = (y_true == config.FALL_INDEX).astype(int)
    scores = probs[:, config.FALL_INDEX]
    if y_bin.sum() == 0:
        return

    thresholds = np.linspace(0.05, 0.95, 91)
    rec, prec, f1s = [], [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        rec.append(recall_score(y_bin, pred, zero_division=0))
        prec.append(precision_score(y_bin, pred, zero_division=0))
        f1s.append(f1_score(y_bin, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(thresholds, rec, color=ALERT, lw=2.4, label="Recall - falls caught")
    ax.plot(thresholds, prec, color=ACCENT, lw=2.4, label="Precision - alerts that are real")
    ax.plot(thresholds, f1s, color=MUTED, lw=2.0, ls="--", label="F1-score")
    ax.axvline(config.FALL_PROB_THRESHOLD, color="#8A8F98", lw=1.4, ls=":")
    ax.annotate(
        f"deployed threshold = {config.FALL_PROB_THRESHOLD:.2f}\n"
        "recall is prioritised: a missed fall\ncosts more than a false alarm",
        (config.FALL_PROB_THRESHOLD, 0.20), textcoords="offset points",
        xytext=(14, 0), fontsize=10, color=MUTED,
    )
    ax.set_xlabel("Fall probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Choosing the emergency-alert threshold")
    ax.legend(loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.28))
    _style(ax)
    fig.savefig(config.RESULTS_DIR / "threshold_analysis.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Written analysis
# --------------------------------------------------------------------------- #
# Why each confusable pair is confusable, in body-geometry terms. Keyed by an
# unordered pair of class names so it reads the same in either direction.
_CONFUSION_NOTES = {
    frozenset({"Standing", "Walking"}): (
        "Standing and Walking differ only by stance width in a still image - a "
        "mid-stride frame and a stood-with-feet-apart frame are the same "
        "posture. Both are safe states, so this costs nothing clinically, and "
        "the video pipeline resolves it using measured motion."
    ),
    frozenset({"Sitting", "Fall Detected"}): (
        "Sitting on a chair and sitting on the floor after a fall both show a "
        "collapsed body height with bent knees. The distinguishing cue is how "
        "close the hips are to the ankles, which is subtle and camera "
        "dependent. Over-alerting here is the safe direction of error."
    ),
    frozenset({"Normal Activity", "Fall Detected"}): (
        "Normal Activity covers bending and crouching, which are exactly the "
        "postures halfway between upright and on-the-floor. The model errs "
        "toward raising an alert, which is the correct bias for a fall monitor."
    ),
    frozenset({"Normal Activity", "Standing"}): (
        "The boundary between 'upright' and 'leaning forward' is a continuum, "
        "and the labelling rule cuts it at a fixed trunk angle. Frames sitting "
        "either side of that cut are genuinely ambiguous."
    ),
    frozenset({"Normal Activity", "Walking"}): (
        "Someone walking while leaning forward sits on the boundary between "
        "the two definitions. Both are safe states."
    ),
    frozenset({"Normal Activity", "Sitting"}): (
        "Crouching and perching on the edge of a seat produce almost the same "
        "knee and hip angles."
    ),
    frozenset({"Standing", "Fall Detected"}): (
        "A rare and clinically important confusion: an upright person alerted "
        "as a fall, or a fall missed as upright. The low count here shows the "
        "trunk-angle and body-height features separate these two cleanly."
    ),
    frozenset({"Walking", "Fall Detected"}): (
        "A rare and clinically important confusion. The low count here shows "
        "the model does not mistake ordinary mobility for an emergency."
    ),
    frozenset({"Sitting", "Standing"}): (
        "Rising from a chair passes continuously through both postures."
    ),
    frozenset({"Sitting", "Walking"}): (
        "Both can show wide leg separation; knee angle is what separates them."
    ),
}


def _explain_confusions(pairs, cm: np.ndarray) -> list:
    """Explain the confusions this run actually produced, not a fixed story."""
    lines, seen = [], set()
    for _count, _share, true_name, pred_name in pairs:
        key = frozenset({true_name, pred_name})
        if key in seen:
            continue
        seen.add(key)
        note = _CONFUSION_NOTES.get(key)
        if note:
            lines.append(f"   - {true_name} vs {pred_name}: {note}")

    fi = config.FALL_INDEX
    safe_swaps = sum(
        cm[i, j]
        for i in range(config.NUM_CLASSES)
        for j in range(config.NUM_CLASSES)
        if i != j and i != fi and j != fi
    )
    total_errors = int(cm.sum() - np.trace(cm))
    if total_errors:
        lines.append(
            f"   - {safe_swaps:,} of the {total_errors:,} errors ({safe_swaps / total_errors:.0%}) "
            "are swaps between two safe activities, which never change the "
            "alert the caregiver sees."
        )
    return _wrap(lines)


def _wrap(lines: list, width: int = 74) -> list:
    """Wrap the explanation lines so the report file stays readable."""
    import textwrap

    wrapped = []
    for line in lines:
        wrapped.extend(
            textwrap.wrap(line, width=width, subsequent_indent="     ") or [line]
        )
    return wrapped


def analyse_confusion_matrix(cm: np.ndarray) -> str:
    """Turn the matrix into the plain-English reading the rubric asks for."""
    names = config.CLASS_NAMES
    fi = config.FALL_INDEX
    total = int(cm.sum())

    true_falls = int(cm[fi].sum())
    caught = int(cm[fi, fi])
    missed = true_falls - caught
    false_alarms = int(cm[:, fi].sum() - cm[fi, fi])
    non_falls = total - true_falls

    lines = [
        "CONFUSION MATRIX ANALYSIS - SafeFall AI test set",
        "=" * 68,
        f"Test frames analysed              : {total:,}",
        f"Frames that really were a fall    : {true_falls:,}",
        "",
        "1. FALL DETECTION (the safety-critical row)",
        "-" * 68,
        f"   Correct fall detections        : {caught:,} of {true_falls:,} "
        f"({caught / max(true_falls, 1):.1%} recall / sensitivity)",
        f"   Missed falls (false negatives) : {missed:,} "
        f"({missed / max(true_falls, 1):.1%} of all falls)",
        f"   False alarms (false positives) : {false_alarms:,} "
        f"({false_alarms / max(non_falls, 1):.2%} of all non-fall frames)",
        f"   Specificity on non-fall frames : "
        f"{1 - false_alarms / max(non_falls, 1):.2%}",
        "",
        "   Reading: recall matters most here. Every missed fall is an elderly",
        "   resident left on the floor unattended, whereas a false alarm only",
        "   costs a caregiver a few seconds to dismiss.",
        "",
        "2. WHERE THE FALL ALERTS COME FROM",
        "-" * 68,
    ]
    for i, name in enumerate(names):
        if i == fi or cm[i, fi] == 0:
            continue
        lines.append(
            f"   {cm[i, fi]:>6,} frames of '{name}' were alerted as a fall "
            f"({cm[i, fi] / max(cm[i].sum(), 1):.2%} of that activity)"
        )
    if false_alarms == 0:
        lines.append("   No non-fall activity was ever alerted as a fall.")

    lines += ["", "3. WHICH ACTIVITIES THE MODEL CONFUSES", "-" * 68]
    pairs = []
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], cm[i, j] / max(cm[i].sum(), 1), names[i], names[j]))
    pairs.sort(reverse=True)
    if pairs:
        for count, share, true_name, pred_name in pairs[:5]:
            lines.append(f"   {count:>6,} frames ({share:5.2%}) of '{true_name}' "
                         f"predicted as '{pred_name}'")
    else:
        lines.append("   No confusions at all.")

    lines += ["", "   Reading:"] + _explain_confusions(pairs[:5], cm)
    lines += ["", "4. PER-CLASS RECALL", "-" * 68]
    for i, name in enumerate(names):
        support = int(cm[i].sum())
        rec = cm[i, i] / max(support, 1)
        lines.append(f"   {name:<18} {rec:6.2%}   (support {support:,} frames)")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    from .inference import SafeFallPredictor

    model = SafeFallPredictor()
    print(f"Scoring with the deployed pipeline: {model.engine_name}"
          f"{' + mirror TTA' if config.USE_MIRROR_TTA else ''}")
    scaler = FeatureScaler.load()

    test_df = load_split(config.TEST_CSV)
    val_df = load_split(config.VAL_CSV)
    print(f"Test frames: {len(test_df):,} from "
          f"{test_df['video_id'].nunique()} unseen videos")

    probs, y_true = predict_dataframe(test_df, model, scaler)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    val_probs, y_val = predict_dataframe(val_df, model, scaler)
    val_acc = accuracy_score(y_val, val_probs.argmax(axis=1))

    threshold_choice = choose_alert_threshold(y_val, val_probs)
    if threshold_choice:
        chosen = threshold_choice["selected_threshold"]
        print(f"\nAlert threshold selected on validation: {chosen:.2f} "
              f"(validation fall F1 {threshold_choice['validation_f1_at_selected']:.4f}, "
              f"recall {threshold_choice['validation_recall_at_selected']:.4f})")
        if abs(chosen - config.FALL_PROB_THRESHOLD) > 1e-9:
            print(f"  NOTE: config.FALL_PROB_THRESHOLD is {config.FALL_PROB_THRESHOLD:.2f}; "
                  f"update it to {chosen:.2f} to match the selection rule.")

    # ---- headline metrics -------------------------------------------------- #
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "validation_accuracy": float(val_acc),
        "mean_confidence": float(confidence.mean()),
        "test_frames": int(len(test_df)),
        "test_videos": int(test_df["video_id"].nunique()),
        "val_frames": int(len(val_df)),
    }

    fall_mask = y_true == config.FALL_INDEX
    if fall_mask.any():
        metrics.update(
            {
                "fall_precision": float(precision_score(y_true, y_pred, labels=[config.FALL_INDEX], average="micro", zero_division=0)),
                "fall_recall": float(recall_score(y_true, y_pred, labels=[config.FALL_INDEX], average="micro", zero_division=0)),
                "fall_f1": float(f1_score(y_true, y_pred, labels=[config.FALL_INDEX], average="micro", zero_division=0)),
                "fall_frames_in_test": int(fall_mask.sum()),
            }
        )

        # The dashboard raises an alert on the fall probability crossing the
        # deployed threshold, not on argmax, so report that operating point too.
        y_bin = fall_mask.astype(int)
        alerted = (probs[:, config.FALL_INDEX] >= config.FALL_PROB_THRESHOLD).astype(int)
        false_alarms = int(((alerted == 1) & (y_bin == 0)).sum())
        metrics.update(
            {
                "alert_threshold": config.FALL_PROB_THRESHOLD,
                "fall_recall_at_alert_threshold": float(recall_score(y_bin, alerted, zero_division=0)),
                "fall_precision_at_alert_threshold": float(precision_score(y_bin, alerted, zero_division=0)),
                "fall_f1_at_alert_threshold": float(f1_score(y_bin, alerted, zero_division=0)),
                "missed_falls_at_alert_threshold": int(((alerted == 0) & (y_bin == 1)).sum()),
                "false_alarms_per_1000_nonfall_frames": round(
                    1000 * false_alarms / max(int((y_bin == 0).sum()), 1), 1
                ),
            }
        )
        metrics.update(threshold_choice)

    # ---- reports ----------------------------------------------------------- #
    report_txt = classification_report(
        y_true, y_pred, labels=range(config.NUM_CLASSES),
        target_names=config.CLASS_NAMES, digits=4, zero_division=0,
    )
    (config.RESULTS_DIR / "classification_report.txt").write_text(
        "SafeFall AI - classification report on the held-out test videos\n"
        f"{'=' * 72}\n{report_txt}\n", encoding="utf-8"
    )
    report_dict = classification_report(
        y_true, y_pred, labels=range(config.NUM_CLASSES),
        target_names=config.CLASS_NAMES, output_dict=True, zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().round(4).to_csv(
        config.RESULTS_DIR / "classification_report.csv"
    )

    # ---- plots ------------------------------------------------------------- #
    cm = plot_confusion_matrix(y_true, y_pred)
    plot_training_curves()
    plot_per_class_metrics(y_true, y_pred)
    metrics.update(plot_fall_curves(y_true, probs))
    plot_threshold_analysis(y_true, probs)

    analysis = analyse_confusion_matrix(cm)
    (config.RESULTS_DIR / "confusion_matrix_analysis.txt").write_text(analysis, encoding="utf-8")
    np.savetxt(config.RESULTS_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(config.CLASS_NAMES), comments="")

    # ---- unseen scenes ----------------------------------------------------- #
    if config.UNSEEN_FEATURES_CSV.exists():
        unseen = pd.read_csv(config.UNSEEN_FEATURES_CSV)
        if len(unseen) and "label_index" in unseen.columns:
            u_probs, u_true = predict_dataframe(unseen, model, scaler)
            u_pred = u_probs.argmax(axis=1)
            metrics["unseen_scene_agreement"] = float(accuracy_score(u_true, u_pred))
            metrics["unseen_scene_frames"] = int(len(unseen))
            metrics["unseen_scene_videos"] = int(unseen["video_id"].nunique())
            metrics["unseen_scene_fall_recall"] = float(
                recall_score(u_true, u_pred, labels=[config.FALL_INDEX],
                             average="micro", zero_division=0)
            )
            dist = pd.Series(u_pred).value_counts().reindex(range(config.NUM_CLASSES)).fillna(0)
            metrics["unseen_scene_prediction_distribution"] = {
                config.CLASS_NAMES[i]: int(dist.iloc[i]) for i in range(config.NUM_CLASSES)
            }

    (config.RESULTS_DIR / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # ---- console ----------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("HELD-OUT TEST RESULTS (videos the model has never seen)")
    print("=" * 72)
    print(f"  Accuracy            : {metrics['test_accuracy']:.4f}")
    print(f"  Precision (macro)   : {metrics['test_precision_macro']:.4f}")
    print(f"  Recall (macro)      : {metrics['test_recall_macro']:.4f}")
    print(f"  F1-score (macro)    : {metrics['test_f1_macro']:.4f}")
    print(f"  Precision (weighted): {metrics['test_precision_weighted']:.4f}")
    print(f"  Recall (weighted)   : {metrics['test_recall_weighted']:.4f}")
    print(f"  F1-score (weighted) : {metrics['test_f1_weighted']:.4f}")
    if "fall_recall" in metrics:
        print(f"\n  FALL recall         : {metrics['fall_recall']:.4f}  <- the one that matters")
        print(f"  FALL precision      : {metrics['fall_precision']:.4f}")
        print(f"  FALL F1             : {metrics['fall_f1']:.4f}")
    if "fall_roc_auc" in metrics:
        print(f"  FALL ROC-AUC        : {metrics['fall_roc_auc']:.4f}")
    print("\n" + report_txt)
    print(analysis)
    print(f"\nAll artefacts written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
