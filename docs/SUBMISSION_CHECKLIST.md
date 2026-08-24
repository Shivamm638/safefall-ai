# FA-2 Submission Checklist — every rubric line and where it is evidenced

The 25 items below are the "FINAL 20/20 SUBMISSION CHECKLIST" from the
assignment brief, in order. Items marked **DONE** are complete in this
repository. Items marked **YOU** need you personally — they cannot be produced
by code.

---

## Status summary

| | Count |
|---|---|
| **DONE** — complete in this repository | 23 of 25 |
| **YOU** — requires you to act | 2 of 25 (deploy the app, record the video) |

---

## The checklist

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | FA-1 dataset/preprocessing work reused and organised | **DONE** | The Le2i dataset from `SafeFall_AI_FA1/Raw_Dataset` is used directly; `src/build_dataset.py` reads it in place. Full breakdown in report §2. |
| 2 | All 5 required activity outputs covered | **DONE** | `Fall Detected`, `Walking`, `Sitting`, `Standing`, `Normal Activity` — defined in `src/config.py`, all five present in train/val/test (see `src/split_dataset.py` output). |
| 3 | Pose estimation implemented and screenshots saved | **DONE** | MediaPipe Pose in `src/pose_utils.py`. Screenshots: `results/pose_estimation_grid.png`, `results/pose_samples/` (10 images). |
| 4 | Classification model implemented and model choice justified | **DONE** | Two-branch CNN in `src/model.py` (127,685 params). Justification in report §4 and on the dashboard's About page. |
| 5 | Clear fall-detection logic explained | **DONE** | Pipeline in report §4.3 and §5.2; implemented in `src/inference.py` (smoothing → 5-frame confirmation → event merging → severity). Diagram on the About page. |
| 6 | Healthcare relevance explained | **DONE** | Report §1 (the "long lie", why wearables fail). Also narrated in `docs/VIDEO_SCRIPT.md` Part 1. |
| 7 | 70 / 15 / 15 train-validation-test split | **DONE** | `src/split_dataset.py` — 91/19/20 videos, **split at video level**, stratified, with a zero-overlap assertion. Report §5.1. |
| 8 | Model trained and saved | **DONE** | `models/fall_detection_cnn.keras`, `models/fall_detection_cnn_weights.npz`, `models/feature_scaler.json`, `models/model_metadata.json`. |
| 9 | Accuracy reported | **DONE** | **83.70 %** test. `results/metrics_summary.json`, `results/classification_report.txt`, dashboard Model Performance page. |
| 10 | Precision reported | **DONE** | **82.05 %** macro / 83.35 % weighted. Same sources. |
| 11 | Recall reported | **DONE** | **79.99 %** macro / 83.70 % weighted; **fall recall 93.47 %** (95.80 % at the deployed alert threshold). |
| 12 | F1-score reported | **DONE** | **79.93 %** macro / 82.74 % weighted; fall F1 0.841. |
| 13 | Confusion matrix generated **and analysed** | **DONE** | `results/confusion_matrix.png` (counts + normalised) and a written reading in `results/confusion_matrix_analysis.txt`, generated from the actual matrix. Report §6.5. |
| 14 | Accuracy graph generated | **DONE** | `results/accuracy_graph.png` — train vs validation per epoch, best epoch annotated. |
| 15 | Loss graph generated | **DONE** | `results/loss_graph.png`. |
| 16 | Prediction screenshots saved | **DONE** | `results/prediction_grid.png`, `results/prediction_samples/` (10 images), `results/fall_sequence.png`, plus `results/misclassification_examples.png` for the error analysis. |
| 17 | Unseen samples tested | **DONE** | Three levels: validation (2,101 frames), test (1,975 frames from 20 unseen videos), and **8,969 frames from 59 videos in two rooms that appear in no split** (Lecture room, Office). Report §6.4. |
| 18 | Real-world challenges / false detections discussed | **DONE** | Report §8 — lighting, camera angle, occlusion, similar postures, false alarms quantified (8.02 % of non-fall frames), dataset realism. Also on the dashboard About page. |
| 19 | Streamlit image upload works | **DONE** | `app.py` Live Monitor → Image upload. Verified end to end. |
| 20 | Streamlit video upload works | **DONE** | `app.py` Live Monitor → Video upload, with timeline, events table and key frames. Verified end to end. |
| 21 | AI prediction works on deployed app | **YOU** | Code is deployment-ready and tested locally. Follow `docs/DEPLOYMENT_GUIDE.md`, then tick this off after testing the live URL. |
| 22 | Prediction confidence displayed | **DONE** | Shown on every result: metric card, probability bar chart across all five classes, per-key-frame captions. |
| 23 | Pose visualisation displayed | **DONE** | Skeleton overlay beside the original for images; key-frame gallery for videos. |
| 24 | Emergency fall alert works | **DONE** | Pulsing red banner, audible alarm (generated in-app, no asset needed), timing, severity and recommended action. |
| 25 | Total activities / fall count / normal count shown | **DONE** | Four metric cards per analysis and on the Monitoring Analytics page. |
| 26 | Analytics / chart included | **DONE** | Activity distribution donut, per-class bar chart, fall-probability timeline, per-second activity ribbon, resident safety index, event log with CSV export. |
| 27 | Dashboard is clean and easy to understand | **DONE** | Four-page healthcare layout, fixed light clinical theme, colour-coded status, plain-English explanation on every prediction. |
| 28 | Future improvements and retraining explained | **DONE** | Report §9 (six improvements + a five-step retraining cycle with fall recall as the release gate). Also on the dashboard About page. |
| 29 | Streamlit Cloud link tested and working | **YOU** | Follow `docs/DEPLOYMENT_GUIDE.md` Step 3 — it has a test checklist. |
| 30 | Final screen recording contains every required evidence item | **YOU** | `docs/VIDEO_SCRIPT.md` gives word-for-word narration for all 12 required items, in the recommended order, with a coverage table at the end. |

---

## What you still have to do

Everything else is finished. These two are yours:

### 1. Deploy the app (~15 minutes)

Follow `docs/DEPLOYMENT_GUIDE.md`. In short: push this folder to a public GitHub
repo, connect it at <https://share.streamlit.io>, set the main file to `app.py`
and the Python version to 3.11 or 3.12, deploy, then test the live URL in a
fresh browser window.

### 2. Record the evidence video (~10 minutes of footage)

Open `docs/VIDEO_SCRIPT.md` beside your screen recorder and read it. It covers
all 12 required items in the recommended order.

The rubric's phrasing is *"Explain instead of only clicking"* — after each result
you show, say one sentence about what it means for an elderly resident. That is
what separates a good video from a full-marks video.

### Optional but worth 5 minutes

Capture stills into a `screenshots/` folder while testing the deployed app: the
dashboard home, a normal prediction, a fall prediction with the alert visible,
the pose output, the metrics page and the confusion matrix. Some markers want
these attached separately to the video.

---

## Marks mapping

| Phase | Marks | Where it is delivered |
|---|---|---|
| Phase 2 — Model selection & fall logic | 5 | Report §4, `src/model.py`, `src/inference.py`, dashboard About page |
| Phase 4 — Evaluation & analysis | 5 | Report §6, `results/` (11 figures + 3 reports), dashboard Model Performance page |
| Phase 5 — Streamlit dashboard | 5 | `app.py` — four pages, uploads, alerts, analytics |
| Phase 8 — Evidence video | 5 | `docs/VIDEO_SCRIPT.md` + your recording + the deployed link |
| **Total** | **20** | |
