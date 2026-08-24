# SafeFall AI — Elderly Fall Detection & Activity Monitoring

**FA-2 — Machine Learning & Deep Learning (20 marks)**
A complete, deployed AI healthcare monitoring system: it estimates human pose,
classifies five activities, detects falls, raises emergency alerts, and presents
everything in a Streamlit dashboard a caregiver can actually read.

| | |
|---|---|
| **Pose estimation** | MediaPipe Pose (BlazePose) — 33 full-body landmarks |
| **Classifier** | Two-branch 1-D CNN, 127,685 parameters |
| **Activity classes** | Fall Detected · Walking · Sitting · Standing · Normal Activity |
| **Dataset** | Le2i Fall Detection (190 videos, 6 rooms, 75,911 frames) |
| **Split** | 70 / 15 / 15 — **at video level**, so no frame leaks between splits |
| **Test accuracy** | **83.70 %** on 1,975 frames from 20 completely unseen videos |
| **Fall recall** | **93.47 %** at argmax — **95.80 %** at the deployed alert threshold |
| **Fall ROC-AUC** | **0.954** |
| **Dashboard** | Streamlit — image upload, video upload, camera snapshot, analytics |

---

## Headline results

Everything below is measured on the **held-out test split**: 20 videos the model
never saw during training or validation.

| Metric | Score |
|---|---|
| Accuracy | 83.70 % |
| Precision (macro) | 82.05 % |
| Recall (macro) | 79.99 % |
| F1-score (macro) | 79.93 % |
| Precision (weighted) | 83.35 % |
| Recall (weighted) | 83.70 % |
| F1-score (weighted) | 82.74 % |

**Per class**

| Activity | Precision | Recall | F1 | Test frames |
|---|---|---|---|---|
| Fall Detected | 0.764 | **0.935** | 0.841 | 429 |
| Normal Activity | 0.709 | 0.395 | 0.508 | 253 |
| Sitting | 0.839 | 0.899 | 0.868 | 325 |
| Standing | 0.893 | 0.908 | 0.901 | 544 |
| Walking | 0.897 | 0.863 | 0.880 | 424 |

**Fall class, treated as the positive class**

| | |
|---|---|
| Recall / sensitivity | 93.47 % — 401 of 429 fall frames caught |
| Precision | 76.38 % |
| F1 | 84.07 % |
| ROC-AUC | 0.954 |
| Specificity on non-fall frames | 91.98 % |
| At the deployed 0.25 alert threshold | recall **95.80 %**, precision 76.38 %, F1 0.841 |

Recall is the number that matters clinically: a missed fall leaves a resident on
the floor, whereas a false alarm costs a caregiver a few seconds. The alert
threshold is therefore not left at a naive 0.5 — it is selected on the
**validation** split by an explicit rule (highest recall among thresholds within
0.5 % of peak F1), which picks 0.25. Across the whole threshold range precision
stays in a narrow band (73.8 %–78.4 %) while recall swings eight points, so buying recall
here is nearly free. See `results/threshold_analysis.png` and
`results/threshold_selection_validation.csv`.

**Generalisation to unseen rooms.** The model was additionally run on 8,969
frames from 59 videos in two rooms (Lecture room, Office) that appear nowhere in
training, validation or test — different furniture, lighting and camera angles.
It agrees with the reference labels on 81.98 % of frames there, essentially the
same as on the test split, so performance is not an artefact of one room.

---

## How it works

```
  Camera frame / uploaded media
        │
        ▼
  [1] MediaPipe Pose  ──►  33 body landmarks (x, y, z, visibility)
        │
        ▼
  [2] Feature engineering
        Branch A: hip-centred, torso-scaled skeleton tensor   (33 × 4)
        Branch B: 25 clinical posture features
                  (trunk angle, knee angle, bounding-box aspect
                   ratio, stance width, body height, …)
        │
        ▼
  [3] Two-branch CNN
        1-D convolutions along the kinematic chain  +  dense fusion head
        │
        ▼
  [4] Activity + confidence
        Fall Detected │ Walking │ Sitting │ Standing │ Normal Activity
        │
        ▼
  [5] Emergency logic
        video : probability smoothing → 5 consecutive confirmed frames
                → event merging → rapid-descent severity check
        image : fall probability above the alert threshold
        │
        ▼
  [6] Caregiver dashboard
        alert · pose overlay · analytics · incident log · CSV export
```

### Why these models

- **MediaPipe Pose** returns 33 full-body landmarks in real time on a plain CPU.
  Working from the skeleton rather than raw pixels makes the system far less
  sensitive to clothing colour, skin tone, furniture and wallpaper, and it is
  inherently more privacy-preserving — which matters a great deal in a care home,
  where continuous video of residents is a serious concern.
- **A convolutional classifier** is the right family because a fall is defined by
  a *local geometric pattern along the body*: the trunk rotating toward
  horizontal, the hips dropping, the legs collapsing. 1-D convolutions along the
  kinematic chain share weights across body parts exactly the way 2-D
  convolutions share them across image patches.
- **The combination** is what makes it clinically usable: pose estimation gives a
  body-centric, lighting-robust representation, and the CNN turns it into an
  activity decision fast enough (~45 frames/second on CPU) to run continuously on
  cheap hardware.

---

## Project structure

```
FA 2 Output/
├── app.py                        Streamlit dashboard (deployment entry point)
├── requirements.txt              runtime dependencies (no TensorFlow — see below)
├── requirements-dev.txt          full training environment
├── packages.txt                  apt packages for Streamlit Cloud
├── .streamlit/config.toml        fixed light clinical theme
│
├── src/
│   ├── config.py                 every path, class, threshold, hyper-parameter
│   ├── pose_utils.py             MediaPipe wrapper + the 25 posture features
│   ├── build_dataset.py          videos → landmarks (multiprocessing)
│   ├── recompute_geometry.py     rebuild features without re-decoding video
│   ├── labeling.py               ground-truth falls + geometric rules
│   ├── split_dataset.py          70/15/15 video-level split
│   ├── augment.py                pose-space data augmentation
│   ├── data.py                   tensors + JSON feature scaler
│   ├── model.py                  the two-branch CNN
│   ├── train.py                  training loop
│   ├── evaluate.py               metrics, confusion matrix, all plots
│   ├── export_model.py           weight export + numerical parity check
│   ├── numpy_inference.py        dependency-free execution engine
│   ├── inference.py              end-to-end pipeline used by the dashboard
│   ├── generate_evidence.py      pose / prediction / failure screenshots
│   └── verify_deployment.py      pre-flight check: runs the app with the
│                                 training-only packages hidden
│
├── notebooks/
│   └── SafeFall_AI_FA2_Pipeline.ipynb    the whole story, runnable
│
├── models/
│   ├── fall_detection_cnn.keras          trained Keras model
│   ├── fall_detection_cnn_weights.npz    exported weights (469 KB)
│   ├── feature_scaler.json               training-time normalisation
│   └── model_metadata.json               model card
│
├── results/
│   ├── confusion_matrix.png              counts + row-normalised
│   ├── accuracy_graph.png                train vs validation accuracy
│   ├── loss_graph.png                    train vs validation loss
│   ├── per_class_metrics.png             precision / recall / F1 per activity
│   ├── fall_detection_curves.png         ROC + precision-recall for falls
│   ├── threshold_analysis.png            why the alert threshold is 0.25
│   ├── fall_sequence.png                 a real fall, frame by frame
│   ├── pose_estimation_grid.png          pose output for all five classes
│   ├── prediction_grid.png               predictions on held-out frames
│   ├── misclassification_examples.png    the errors, and why they happen
│   ├── classification_report.txt / .csv
│   ├── confusion_matrix_analysis.txt     the matrix read in plain English
│   ├── metrics_summary.json
│   ├── engine_parity_check.json
│   └── pose_samples/ , prediction_samples/
│
├── data/
│   ├── processed/   pose feature tables (regenerated, git-ignored)
│   ├── splits/      train / val / test tables
│   └── samples/     small demo clips + stills bundled with the app
│
└── docs/
    ├── PROJECT_REPORT.md         the full write-up
    ├── VIDEO_SCRIPT.md           word-for-word narration for the evidence video
    ├── DEPLOYMENT_GUIDE.md       Streamlit Cloud, step by step
    └── SUBMISSION_CHECKLIST.md   every rubric line, and where it is evidenced
```

---

## Running it

### The dashboard

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

> **Use `python -m streamlit`, not a bare `streamlit`.** On Windows the Microsoft
> Store build of Python installs console scripts into
> `…\LocalCache\local-packages\Python312\Scripts`, which is **not** added to
> `PATH`. Typing `streamlit run app.py` there fails with
> *"The term 'streamlit' is not recognized as the name of a cmdlet…"* even though
> Streamlit is installed correctly. Running it as a module bypasses `PATH`
> entirely and always works. The same applies to `jupyter` — use
> `python -m jupyter notebook`.
>
> To get the short form working permanently, add the Scripts folder to your user
> `PATH` once, in PowerShell (then reopen the terminal):
>
> ```powershell
> $s = python -c "import sysconfig,os; print(sysconfig.get_path('scripts', f'{os.name}_user'))"
> [Environment]::SetEnvironmentVariable('PATH', "$([Environment]::GetEnvironmentVariable('PATH','User'));$s", 'User')
> ```

The app ships with sample clips and stills, so it can be demonstrated
immediately without hunting for a fall video — open **Live Monitor → Video
upload → "No clip to hand? Try a bundled sample"**.

### Reproducing the whole pipeline

```bash
pip install -r requirements-dev.txt
```

Point the pipeline at your copy of the Le2i dataset (the raw videos are ~17 GB
and are not in this repository):

```bash
export SAFEFALL_RAW_DATASET="/path/to/Raw_Dataset"
```

Then run the stages in order:

```bash
python -m src.build_dataset      # MediaPipe pose extraction (~13 min, 10 workers)
python -m src.labeling           # ground-truth falls + geometric activity rules
python -m src.split_dataset      # 70/15/15 by video
python -m src.train              # ~100 s on CPU
python -m src.export_model       # NumPy weights + parity check
python -m src.evaluate           # metrics + every plot
python -m src.generate_evidence  # screenshots + demo media
```

### Before deploying

```bash
python -m src.verify_deployment
```

This hides every training-only package (TensorFlow, scikit-learn, seaborn,
joblib) and then drives the whole app — model load, pose estimation, image
prediction, video analysis, alert logic — to prove the dashboard runs on
`requirements.txt` alone. It caught a real bug during development: MediaPipe's
drawing utilities import matplotlib at module load, so the pose overlay had a
dependency nothing had declared.

---

## Two engineering decisions worth explaining

### 1. The split is made at video level, not frame level

Neighbouring frames of the same recording are nearly identical. A random
frame-wise split would put near-duplicates on both sides of the fence and report
an accuracy that could never be reproduced on a new resident. Splitting whole
videos — stratified by room and by whether the video contains a fall — keeps the
test set genuinely unseen. It reports a lower number than a frame-wise split
would, and it is the only number that means anything.

### 2. The deployed app runs the trained network without TensorFlow

TensorFlow is a ~600 MB dependency, and Streamlit Community Cloud gives an app
1 GB of RAM — which TensorFlow, MediaPipe and OpenCV together will not reliably
fit inside. Rather than shrink the model or give up on deployment, the trained
weights are exported and the identical architecture is re-implemented in pure
NumPy (`src/numpy_inference.py`).

It is the same network doing the same arithmetic, and this is verified rather
than asserted — `python -m src.export_model` compares both engines across the
whole test split and refuses to ship if they disagree:

```
  samples compared          : 1,975
  max abs probability diff  : 1.013e-06
  identical predicted class : 100.0000%
  PASSED
```

The deployed app therefore runs the real trained model on ~40 MB of
dependencies instead of ~700 MB.

---

## Honest limitations

- **Weak labels for four of the five classes.** Falls come from the Le2i
  ground-truth annotation files — no frame in the training set is ever called a
  fall on posture evidence alone. But Le2i does not annotate Walking, Sitting,
  Standing or Normal Activity, so those four are derived from a transparent,
  published rule set over the same geometric features the network sees
  (`src/labeling.py`). The classifier therefore inherits the boundaries those
  rules draw, and its accuracy on those four classes partly measures how well it
  reproduces an explainable rule set. **The fall-class metrics are the ones
  measured against human annotation, and they are the ones to judge the system
  on.**
- **Single-frame ambiguity.** Walking and Standing differ by motion, which one
  still image cannot show. The classes are therefore defined by *stance* so the
  task is well-posed for a single frame, and the video pipeline corrects the call
  using measured hip motion once several frames exist.
- **Normal Activity is the weakest class (0.53 F1).** It covers bending,
  reaching and crouching — exactly the postures halfway between upright and
  on-the-floor — so it sits on every decision boundary at once. 54 % of all the
  model's errors are swaps between two *safe* activities that never change the
  alert a caregiver sees.
- **Occlusion.** If furniture hides the hips or legs, MediaPipe returns a partial
  skeleton and the system reports "no person detected" rather than guessing. In
  this dataset a usable skeleton was found in 84.8 % of frames.
- **Low light.** Pose detection rate drops in poorly lit rooms; an infrared
  camera would be the practical fix.
- **Dataset scope.** Le2i uses staged falls performed by younger actors in six
  rooms. Real elderly falls are slower, more varied and often partly occluded, so
  these numbers should be treated as an upper bound on real-world performance
  until validated on genuine care-home footage.

---

## Dataset credit

Le2i Fall Detection Dataset — Laboratoire Electronique, Informatique et Image,
Université de Bourgogne.

> I. Charfi, J. Miteran, J. Dubois, M. Atri, R. Tourki, "Optimised
> spatio-temporal descriptors for real-time fall detection: comparison of SVM and
> Adaboost based classification", *Journal of Electronic Imaging (JEI)*, Vol. 22,
> Issue 4, October 2013.
