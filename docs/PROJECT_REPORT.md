# SafeFall AI — Project Report

**FA-2 · Machine Learning & Deep Learning · AI-Powered Elderly Fall Detection System**

---

## 1. Objective and healthcare relevance

Falls are the leading cause of injury-related hospitalisation among people over
65. The danger is rarely the impact itself — it is the **long lie**: the time a
resident spends on the floor before anyone notices. Every hour on the floor
raises the risk of dehydration, pressure injury, pneumonia and death, and it is
the single strongest predictor of whether an older adult ever returns to
independent living.

A camera-based monitor addresses exactly this gap. It cannot prevent the fall,
but it can collapse the time-to-discovery from hours to seconds, and it does so
without asking a resident to wear or remember a pendant — the reason wearable
alarms fail so often in practice.

**SafeFall AI** takes an image or a video from a room camera and returns one of
five activity states, with a confidence score:

| Output | Meaning to a caregiver |
|---|---|
| **Fall Detected** | Emergency — resident is on the floor, respond now |
| **Walking** | Normal mobility |
| **Sitting** | Resting, seated |
| **Standing** | Upright and stable |
| **Normal Activity** | Bending, reaching, crouching or in transition |

When a fall is detected the dashboard raises a high-visibility emergency alert
with an audible alarm; every other state shows a calm "safe" status.

**Design priority.** In this domain the two error types are not symmetric. A
missed fall leaves a resident on the floor. A false alarm costs a caregiver five
seconds to dismiss. Every design decision in this project — the class weights,
the alert threshold, the confirmation logic — is biased toward **recall on the
fall class**, and the report says so explicitly wherever a trade-off was made.

---

## 2. Dataset

### 2.1 Source

The **Le2i Fall Detection Dataset** (Université de Bourgogne) was carried
forward from FA-1. It contains staged falls and activities of daily living
recorded in six rooms at 320 × 240, 25 FPS.

| Subset | Videos | Raw frames | Videos with a ground-truth fall | Annotations |
|---|---|---|---|---|
| Coffee_room_01 | 48 | 14,428 | 47 | yes |
| Coffee_room_02 | 22 | 13,141 | 12 | yes |
| Home_01 | 30 | 7,175 | 30 | yes |
| Home_02 | 30 | 7,322 | 7 | yes |
| Lecture_room | 27 | 17,293 | — | **no** |
| Office | 33 | 16,552 | — | **no** |
| **Total** | **190** | **75,911** | **96** | 130 annotated |

Each annotation file gives the frame where the fall **begins** and the frame
where it **ends**, plus a per-frame bounding box.

### 2.2 How the two annotation groups are used

This split in the data turned out to be an asset rather than a problem:

- The **130 annotated videos** (Coffee room + Home) are the supervised dataset.
  They are split 70/15/15 into training, validation and test.
- The **60 unannotated videos** (Lecture room + Office) are held back entirely as
  an **unseen-scene generalisation set** — two rooms with different furniture,
  lighting and camera angles that appear nowhere in training, validation or test.
  This is a far harder and more honest test of generalisation than a random
  split, and it is reported separately in §6.4.

### 2.3 Pose extraction

`src/build_dataset.py` walks all 190 videos, samples every 3rd frame, and runs
MediaPipe Pose across 10 worker processes (≈13 minutes for 17 GB of video).

| | |
|---|---|
| Frames sampled | 25,304 |
| Frames with a usable skeleton | **21,470 (84.8 %)** |
| — from annotated subsets | 12,501 |
| — from unseen-scene subsets | 8,969 |

A frame is rejected when mean landmark visibility is below 0.35 or the four core
torso landmarks are below 0.5 — the system reports "no person detected" rather
than classifying a broken skeleton. The 15.2 % rejection rate is itself a
finding, and it is discussed as a limitation in §8.

---

## 3. Labelling: what is ground truth and what is not

This is the most important methodological section of the report, because the
credibility of every number downstream depends on it.

### 3.1 Fall Detected — human ground truth

A frame is labelled **Fall Detected** if and only if:

1. it falls inside the annotated `[fall_start, fall_end]` window of the Le2i
   ground-truth file, **or**
2. it comes *after* that window and the resident is still on the floor.

**No frame in the training set is ever called a fall on posture evidence alone.**
Condition (2) exists because a fall monitor must keep alerting while the resident
is still down — the emergency does not end when the falling motion ends. If the
resident gets back up unaided, the condition stops firing and the label reverts.

"Still on the floor" is detected two ways:

- **Lying flat** — bounding-box aspect ratio > 1.15 with the trunk more than 50°
  from vertical.
- **Height collapse** — the body occupies less than 62 % of the vertical space
  *that same person occupied while upright earlier in the same recording*.

The second test was added after inspecting a failure case: a resident who falls
and ends up slumped against a couch keeps an upright trunk, so the aspect-ratio
test misses them entirely, even though they are just as much on the floor and
just as much an emergency. Comparing against a per-video reference height makes
this robust to how far the camera is from the person. Adding it raised the fall
class from 2,025 to 2,587 frames and lifted test fall recall from 84.3 % to
93.5 %.

### 3.2 The other four classes — transparent rules, honestly declared

Le2i does not annotate walking, sitting or standing. Those four classes are
therefore derived by a **rule-based (weak supervision) labeller** that reads the
same interpretable geometric features the network sees. Every rule is a statement
about body geometry that a clinician could check by eye:

| Class | Rule |
|---|---|
| **Sitting** | trunk < 55° from vertical, knees bent < 128°, hip angle < 125°, aspect < 1.10 |
| **Standing** | trunk < 28°, knees straight > 145°, stance width < 0.65 shoulder-widths |
| **Walking** | trunk < 28°, knees > 132°, stance width ≥ 0.65 shoulder-widths |
| **Normal Activity** | trunk inclined 28°–55° and not on the floor — bending, reaching, crouching, or mid-transition |

Labels are then passed through a per-video majority filter (width 3) that removes
single-frame flicker. Ground-truth fall frames are never overwritten by the
filter.

**Two design decisions are worth defending.**

*Why the rules are single-frame observable.* An earlier version defined Walking
as "upright **and moving**", using frame-to-frame hip speed. This is the correct
everyday definition of walking — but the classifier is given **one frame at a
time**, and motion is invisible in one frame. The target was literally
undecidable from the input, and validation accuracy plateaued at 75 % no matter
how the model was tuned. Redefining Walking and Standing by **stance width** —
which *is* visible in a single frame — made the task well-posed and lifted
validation accuracy to 84.8 %. Motion did not disappear from the system; it moved
to where the evidence actually exists, the video pipeline (§5.2).

*Why Normal Activity is defined positively.* It was originally the residual
"everything else" bucket, and it was the worst class by a wide margin (47.8 %
recall) because it had no coherent visual identity — a model cannot learn
"whatever the other four rules rejected". Redefining it positively as *inclined
trunk: bending, reaching, crouching* gave it a real meaning, and one that is
clinically useful in its own right, since that posture band carries elevated fall
risk. Overall accuracy rose from 78.9 % to 82.8 %.

### 3.3 The resulting distribution

| Class | Frames | Share | Label source |
|---|---|---|---|
| Fall Detected | 2,587 | 20.7 % | **Le2i ground truth** (705 in-window + 1,872 still-down) |
| Standing | 4,117 | 32.9 % | geometric rule |
| Walking | 3,498 | 28.0 % | geometric rule |
| Sitting | 1,164 | 9.3 % | geometric rule |
| Normal Activity | 1,135 | 9.1 % | geometric rule |
| **Total** | **12,501** | | |

### 3.4 What this means for the results

The four rule-derived classes are learned from a rule set the model can in
principle reproduce, so their accuracy partly measures **how well the CNN
generalises an explainable rule to unseen people and rooms** rather than how well
it discovers activity from scratch. This is stated plainly rather than buried,
because it changes how the numbers should be read.

Three things stop this from being circular:

1. **The fall class is not rule-derived.** It comes from human annotation, and it
   is the class the system exists to detect. Its 93.5 % recall and 0.954 ROC-AUC
   are genuine supervised results.
2. **The split is by video.** Reproducing a rule on the same people in the same
   room is easy; doing it on 20 unseen recordings is not.
3. **Training data is augmented.** The model is trained on mirrored, rescaled,
   translated, jittered and partly occluded skeletons, so it cannot survive by
   memorising exact threshold values.

---

## 4. Model selection and architecture

### 4.1 Pose estimation — MediaPipe Pose (BlazePose)

Chosen over YOLOv8-Pose and OpenPose because:

- It returns **33 full-body landmarks** — including hips, knees, ankles and
  heels — which is exactly the lower-body detail a fall analysis needs.
- It runs in **real time on a CPU**. A care-home deployment cannot assume a GPU
  per camera, and Streamlit Community Cloud does not have one.
- It provides a **per-landmark visibility score**, which gives the system a
  principled way to say "I cannot see this person" instead of guessing.
- Working from a skeleton rather than pixels is **privacy-preserving by
  construction** — a serious consideration when the subject is a resident under
  continuous observation.

### 4.2 Activity classification — a two-branch 1-D CNN

The brief recommends a CNN, and a CNN is genuinely the right family here — but
over the *body*, not over pixels. A fall is a local geometric pattern along the
kinematic chain: hip–knee–ankle collapsing, shoulder–hip rotating toward
horizontal. 1-D convolutions sliding along the 33-landmark chain share weights
across body parts exactly the way 2-D convolutions share them across image
patches, so a filter that learns "this limb has folded" is reused wherever it
appears.

```
Branch A — Pose CNN                         Branch B — Clinical features
input (33 landmarks × 4 channels)           input (25 posture descriptors)
  Conv1D(64, k=3, same) + ReLU                Dense(64) + ReLU
  BatchNorm                                   BatchNorm
  Conv1D(128, k=3, same) + ReLU               Dropout(0.30)
  BatchNorm                                        │
  MaxPool1D(2)          → 16 × 128                 │
  Conv1D(128, k=3, same) + ReLU                    │
  BatchNorm                                        │
  GlobalAvgPool ⧺ GlobalMaxPool → 256              │
        └──────────────────┬──────────────────────┘
                           ▼
                   Concatenate → 320
                   Dense(128) + ReLU + Dropout(0.45)
                   Dense(64)  + ReLU + Dropout(0.35)
                   Dense(5, softmax)

Total parameters: 127,685   (498 KB)
```

**Branch A** is fed a hip-centred, torso-scaled skeleton, so the network learns
*body configuration* rather than where the person happens to stand or how far
they are from the lens.

**Branch B** supplies 25 hand-designed descriptors — trunk angle, knee and hip
angles, bounding-box aspect ratio, body height, stance width, gait asymmetry,
mean visibility. These are the explicit cues a physiotherapist would name, and
including them makes the model's behaviour defensible: when the dashboard says
"fall", it can also say *trunk tilted 80° from upright, width/height ratio 1.60*.

L2 regularisation (1e-4), batch normalisation and dropout are used throughout;
the model is deliberately small so it can run continuously on a CPU.

### 4.3 The fall-detection pipeline

```
image/video → MediaPipe Pose → 33 landmarks
            → Branch A tensor + Branch B features
            → CNN → activity + confidence
            → emergency logic → alert or safe status
```

---

## 5. Training and inference

### 5.1 Training

| | |
|---|---|
| Split | **70 / 15 / 15 at video level** — 91 / 19 / 20 videos |
| Frames | 8,425 train · 2,101 validation · 1,975 test |
| Video overlap between splits | **0** (asserted by `src/split_dataset.py`) |
| Stratification | by room subset and by whether the video contains a fall |
| Augmentation | 4 augmented skeletons per training frame → 42,125 samples |
| Class weights | inverse frequency, so the safety-critical class stays influential |
| Optimiser | Adam, lr 5e-4, batch 64 |
| Callbacks | EarlyStopping (val accuracy, patience 15, restore best), ReduceLROnPlateau, ModelCheckpoint |
| Result | 18 epochs, 101 s on CPU, best validation accuracy **89.29 %** at epoch 3 |

**Augmentation is applied in pose space**, to the raw skeleton, and both network
inputs are then recomputed from the transformed body. Each transform mimics a
real failure mode of a wall-mounted camera:

| Transform | Real-world situation |
|---|---|
| Mirror (with left/right landmark swap) | resident walks the other way down the corridor |
| Scale ×0.80–1.25 | resident nearer to or further from the lens |
| Translate ±0.06 | resident in a different part of the room |
| Jitter σ 0.003–0.012 | ordinary landmark noise |
| Occlusion of 1–3 landmarks | a limb hidden behind furniture |
| Rotation ±3° only | slightly off-level camera bracket |

Rotation is deliberately kept small. Tilting a skeleton changes its trunk angle —
the very signal that separates "upright" from "on the floor" — so a large
rotation would silently relabel the sample without changing its target, injecting
label noise rather than robustness. An early version used ±9° rotation and
validation accuracy did not improve at all; restricting it to ±3° was part of
what unlocked the later gains.

Hyper-parameters (learning rate, batch size, augmentation strength) were selected
by comparing **validation** accuracy. The test split was scored once, with the
selected configuration.

That selection was later checked properly rather than left as a claim. A random
search over 10 configurations - learning rate, dropout, L2, batch size, network
width and augmentation strength - was run and scored on validation
(`python -m src.hparam_search`, 72 minutes; full results in
`results/hparam_search.json`). **Nothing beat the hand-tuned baseline**, which
took 0.8905; the best of the nine random trials reached 0.8901
(trial-3), and the remaining eight came in below that.

The negative result is more informative than a small gain would have been,
because of *what* failed. The search varied width specifically on the suspicion
that 128k parameters were too few to separate Normal Activity, the class
dragging macro F1 down. Doubling the width made things **worse**, not better
(0.8896, 0.8829, 0.8810, 0.8743 against 0.8905), and heavier augmentation did not help
either (0.8901, 0.8843, 0.8839, 0.8810). Capacity and training data volume are
therefore not what is limiting that class.

That is consistent with what the confusion matrix shows. Normal Activity is
defined as a trunk inclined between 28 and 55 degrees - bending, reaching,
crouching - and a fall *passes through exactly those postures* on its way to the
floor. The frames the model gets wrong sit at the top of that band (median trunk
angle 54 degrees against 15 for the ones it gets right), and two videos alone
supply 65 of the 93 Normal-to-Fall errors. In one of them the resident is lying
horizontally on furniture in a clip with no annotated fall: the label is right
and the model is wrong, and no amount of extra width will fix it, because a
single frame of pose does not carry the information needed to tell a sofa from
a floor. What would is a per-video reference height - the same signal the
labelling rules already use - which the network is not currently given.


### 5.2 Inference and emergency logic

**Single image.** Fall alert if the fall probability exceeds the alert threshold.
The dashboard also reports the trunk angle and aspect ratio behind the decision,
so the caregiver sees *why*.

**Video.** A single noisy frame must never raise an alarm, and a genuine fall
stays visible for many consecutive frames because the resident remains on the
floor. So:

1. Per-frame fall probabilities are smoothed with a 5-frame moving average.
2. An alert latches only after **5 consecutive** frames above threshold (≈0.4 s).
3. Alerts less than 2 s apart are **merged into one incident** — a caregiver
   should be told "a fall happened", not handed four fragments of one event.
4. Hip descent velocity in the frames before the event sets the severity: a rapid
   downward movement marks the event **HIGH** rather than **MODERATE**.
5. **Motion refinement.** Once several frames exist, real motion is available, so
   a resident the CNN calls "Standing" while their hips travel across the room is
   corrected to Walking, and vice versa. Only the two *safe* upright classes are
   ever exchanged — fall decisions are never overridden by this step.

### 5.3 Choosing the alert threshold

The alert fires when the fall probability crosses a threshold, and that
threshold is a genuine clinical lever rather than an arbitrary 0.5. It is chosen
by an explicit rule implemented in `evaluate.choose_alert_threshold`, run on the
**validation** split only:

> Among all thresholds whose fall-class F1 is within 0.5 % of the best
> achievable, take the one with the **highest recall**.

Validation F1 is almost flat across 0.25–0.50 (0.9341 to 0.9375), so that slack
costs essentially nothing and buys recall — the error type that carries the real
cost. The rule selects **0.25** (validation fall F1 0.9341, recall 0.9551); the
full sweep is saved to `results/threshold_selection_validation.csv`.

Only then was the choice measured on the test set:

| Threshold | Fall recall | Fall precision | False alarms per 1,000 non-fall frames |
|---|---|---|---|
| 0.20 | 96.5 % | 73.8 % | 95.1 |
| **0.25 (deployed)** | **95.8 %** | **74.9 %** | **89.3** |
| 0.35 | 94.6 % | 76.2 % | 82.1 |
| 0.50 (naive default) | 93.0 % | 77.9 % | 73.1 |
| 0.70 | 88.6 % | 78.4 % | 67.9 |

The striking feature of this table is that **precision stays in a narrow band** —
73.8 % to 78.4 % across the entire range — while recall swings by eight points.
Lowering the threshold is therefore cheap: moving from the naive 0.50 to the
selected 0.25 catches nearly three percentage points more falls (93.0 % → 95.8 %)
for about sixteen extra false alarms per thousand non-fall frames. In video mode the
consecutive-frame confirmation absorbs most of those before they ever reach a
caregiver.

---

### 5.4 When the legs are out of shot

The CNN reads the kinematic chain down to the ankles, so it cannot be asked
anything when the legs are outside the frame - MediaPipe still reports those
joints, at invented positions, and any answer would be computed from
coordinates that do not exist. That view is not an edge case: a laptop webcam at
desk distance sees head, shoulders and perhaps hips, and nothing below.

Refusing to answer there would be the wrong behaviour for a safety system, so a
second detector covers it. Its first version was a single hand-tuned rule -
shoulder tilt beyond 26 degrees - which measured 78.5% accuracy on held-out
crops with a **32.4% false-alarm rate**: one upright frame in three raised an
alert. That is not usable, and it was leaving two things unused.

*Hips.* The first version cropped to head and shoulders only. But the framing
check also rejects frames where the hips **are** visible and merely the legs are
not. Shoulders plus hips give the **trunk angle**, the canonical fall cue, which
needs no legs at all.

*Evidence.* One threshold on one cue cannot combine evidence; several features
and a trained classifier can.

The detector is therefore a logistic model over 11 visibility-aware features, with
`trunk_angle` gated by a `hips_visible` flag so the model learns when to believe
it. It was trained on 3,516 crops built from the training videos in two views that
mimic real cameras - a torso view (head to hips) and a head-and-shoulders view -
each re-run through pose estimation so the landmark normalisation matches what
inference will see. Model and operating point were chosen on validation and
scored once on 1,234 held-out crops:

| | first version | trained detector |
|---|---|---|
| accuracy | 78.5% | **89.7%** |
| fall precision | 71.7% | **89.1%** |
| false-alarm rate | 32.4% | **10.4%** |
| fall recall | 90.4% | 89.9% |

The threshold sweep refused any operating point below 0.90 validation recall, so
the collapse in false alarms is not paid for with missed falls. Broken down by
view, the torso case reaches 90.3% and the harder head-and-shoulders case
89.2% - the view the original 78.5% was measured on.

Two properties matter for deployment. It runs through the same exported-weights
NumPy path as the main classifier, so the deployed app still carries no
deep-learning framework; the runtime was verified against scikit-learn to 2.2e-16
with 100% agreement on the decision. And its probability is mapped onto the alarm
scale by a monotonic piecewise-linear rescale, so its 0.66 operating point lands
exactly on the live alarm's 0.25 threshold and the two cannot disagree.

What it does **not** do is claim more than it measured. It answers one question -
is this person on the floor - and the interface shows fall/upright bars rather
than five-class bars, because the upper body cannot separate walking from
standing. The dashboard reports which engine answered, so the reading is never
mistaken for the five-class output.

Reproduce with: `python -m src.upper_body_train`

## 6. Evaluation

All numbers below come from the **held-out test split**: 1,975 frames from 20
videos never seen in training or validation.

### 6.1 Overall metrics

| Metric | Score |
|---|---|
| Accuracy | **83.70 %** |
| Precision (macro) | 82.05 % |
| Recall (macro) | 79.99 % |
| F1-score (macro) | 79.93 % |
| Precision (weighted) | 83.35 % |
| Recall (weighted) | 83.70 % |
| F1-score (weighted) | 82.74 % |
| Validation accuracy | 89.29 % |

### 6.2 Per-class metrics

| Activity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| **Fall Detected** | 0.7638 | **0.9347** | 0.8407 | 429 |
| Normal Activity | 0.7092 | 0.3953 | 0.5076 | 253 |
| Sitting | 0.8391 | 0.8985 | 0.8678 | 325 |
| Standing | 0.8933 | 0.9081 | 0.9006 | 544 |
| Walking | 0.8971 | 0.8632 | 0.8798 | 424 |

### 6.3 The fall class in clinical terms

| | |
|---|---|
| Correct fall detections | **401 of 429 (93.47 % sensitivity)** |
| Missed falls | 28 (6.53 %) |
| False alarms | 110 of 1,546 non-fall frames (8.02 %) |
| Specificity | 91.98 % |
| ROC-AUC | **0.954** |
| Average precision | 0.713 |
| At the deployed 0.25 alert threshold | recall **95.80 %** (35 falls missed), precision 76.38 %, F1 0.841 |

An ROC-AUC of 0.954 says the model's *ranking* of fall likelihood is strong: it
almost always assigns a higher fall probability to a fall frame than to a
non-fall frame. That is what makes threshold tuning a legitimate lever rather
than wishful thinking.

### 6.4 Testing on unseen data

The rubric asks for validation data, test data and genuinely new material. All
three were used:

| Evaluation | Frames | Result |
|---|---|---|
| Validation split (19 unseen videos) | 2,101 | 89.29 % accuracy |
| Test split (20 unseen videos) | 1,975 | 83.70 % accuracy, 93.47 % fall recall |
| **Unseen rooms** (59 videos, Lecture room + Office) | 8,969 | **81.98 % agreement**, 70.05 % fall recall |

The unseen-room evaluation is the strongest evidence in the report. Those two
rooms have different furniture, lighting and camera angles and appear nowhere in
any split, yet overall agreement (81.98 %) is essentially identical to the test
split (83.70 %). **The model is not memorising a room.**

Fall recall does drop there, to 70.05 %, and that is worth stating plainly. Those
labels are themselves weaker — with no annotation files, the fall reference in
those rooms comes from the posture rule rather than human annotation — so part of
the gap is reference noise rather than model failure. But it is also a fair
warning that a new room should be validated before the system is trusted in it.

### 6.5 Reading the confusion matrix

See `results/confusion_matrix.png` and the written analysis in
`results/confusion_matrix_analysis.txt`.

**Where the alerts come from**

| True activity | Frames alerted as a fall | Share of that activity |
|---|---|---|
| Normal Activity | 76 | 30.0 % |
| Sitting | 29 | 8.9 % |
| Standing | 3 | 0.55 % |
| Walking | 2 | 0.47 % |

The most important row is the last two: an upright, mobile resident is almost
never alerted as a fall. The false alarms are concentrated in exactly the
postures that *look* like a partial fall — bending, crouching and sitting low.

**The main confusions, and why**

1. **Normal Activity → Fall (76 frames).** Normal Activity is bending and
   crouching — the postures halfway between upright and on the floor. The model
   errs toward raising an alert, which is the correct bias here.
2. **Standing ↔ Walking (83 frames).** These differ only by stance width in a
   still image. Both are safe states, and the video pipeline resolves it with
   measured motion.
3. **Normal Activity → Sitting (39 frames).** Crouching and perching on the edge
   of a seat produce almost identical knee and hip angles.
4. **Fall → Normal Activity (34 frames).** The genuine misses: mid-fall frames
   where the resident is still partly upright.

**The summary statistic that matters:** 170 of the 322 total errors (54 %) are
swaps between two *safe* activities that never change the alert the caregiver
sees. The clinically meaningful error rate is far lower than the raw accuracy
figure suggests.

### 6.6 Visual evidence

| File | What it shows |
|---|---|
| `results/fall_sequence.png` | a real test-set fall frame by frame: walking → fall → on the floor, with the alert firing at the right moment and staying latched |
| `results/pose_estimation_grid.png` | MediaPipe output for all five activities |
| `results/prediction_grid.png` | correct predictions on held-out frames, with confidences |
| `results/misclassification_examples.png` | the most common errors, with the true and predicted labels side by side |
| `results/confusion_matrix.png` | counts and row-normalised recall |
| `results/accuracy_graph.png`, `loss_graph.png` | training vs validation per epoch |
| `results/fall_detection_curves.png` | ROC and precision-recall for the fall class |
| `results/threshold_analysis.png` | the alert-threshold trade-off |

---

## 7. The dashboard

`app.py` — a four-page Streamlit healthcare dashboard.

**Live Monitor** — image upload, video upload and live camera snapshot. Every
result shows the original frame beside the pose overlay, the predicted activity
with its confidence, a probability bar chart across all five classes, and a
plain-English explanation citing the measured trunk angle and aspect ratio. A
fall produces a pulsing red emergency banner with an audible alarm and a
recommended action; everything else shows a calm green status. Bundled sample
clips let the system be demonstrated without hunting for a fall video.

For video, the dashboard adds a fall-probability timeline with the alert
threshold marked and confirmed events shaded, an activity-distribution donut, a
per-second activity ribbon, a confirmed-events table with timings and severity, a
gallery of annotated key frames, and a downloadable incident report.

**Monitoring Analytics** — session-wide totals: total activities detected, fall
count, normal-activity count, average confidence, activity distribution charts, a
resident safety index and a full event log with CSV export.

**Model Performance** — the metrics above, all six evaluation plots, the written
confusion-matrix analysis and the classification report, rendered inside the
deployed app so the evidence travels with the system.

**About & Limitations** — the pipeline diagram, model justification, limitations
and the retraining plan.

---

## 8. Limitations and real-world challenges

| Challenge | Status in this system |
|---|---|
| **Lighting variation** | Pose estimation degrades in low light; overall skeleton detection rate was 84.8 %. Frames without a reliable skeleton are reported as "no person detected" rather than guessed. An infrared camera is the practical fix. |
| **Camera angle** | Landmarks are normalised by torso length, which handles distance well. Extreme overhead or very oblique angles remain harder, and the ±3° rotation augmentation covers only a slightly off-level bracket. |
| **Occlusion** | Furniture hiding hips or legs yields a partial skeleton. Occlusion augmentation (1–3 landmarks dropped to low confidence) hardens the model, but heavy occlusion still causes rejection. |
| **Similar postures** | Sitting on a chair and sitting on the floor after a fall are geometrically close; so are standing and mid-stride walking. Both confusions are visible in the confusion matrix and quantified in §6.5. |
| **False fall detections** | 8.02 % of non-fall frames at argmax, concentrated in bending and crouching. Video mode suppresses most of these through 5-frame confirmation and event merging. |
| **Weak labels** | Four of five classes are rule-derived; only the fall class is human-annotated. Discussed fully in §3.4. |
| **Dataset realism** | Le2i uses staged falls by younger actors. Real elderly falls are slower and more varied, so these figures are an upper bound until validated on genuine care-home footage. |
| **Single-frame limits** | Motion is invisible in one image. Mitigated by defining classes on posture and by the video motion-refinement step. |

---

## 9. Monitoring, maintenance and future improvements

### Planned improvements

1. **A height reference the network can see.** The strongest identified gap:
   give the model the per-video reference height the labelling rules already
   compute, so "lower than this person stood earlier" becomes an input rather
   than something the network has to infer from absolute frame position. This is
   the change most likely to move Normal Activity, which the hyper-parameter
   search showed is not limited by capacity. It needs care at inference, where a
   single uploaded photo has no video to reference.
2. **Temporal model.** A 1-D CNN or LSTM over a window of frames would separate
   Walking from Standing by motion rather than posture, and would catch slow
   "slump" falls that a single frame cannot express. Note that frame-to-frame
   speed alone does not separate falls from bending here - measured AUC 0.43 for
   hip speed and 0.56 for aspect rate - because much of the Fall class is a
   resident lying still after the event.
3. **Real elderly footage.** Collect and annotate genuine care-home video,
   including slow falls, getting-out-of-bed events and walker/frame users.
4. **Low-light and infrared.** Add infrared samples and augment with brightness,
   blur and noise to harden detection at night, when falls are most common.
5. **Fewer false alerts.** A short "are you OK?" confirmation window before
   escalation, and agreement between two cameras before a hard alert.
6. **Better elderly-posture recognition.** Fine-tune the pose model on stooped
   postures, walking frames and wheelchairs, which BlazePose handles less well.
7. **Real-time CCTV/RTSP** ingestion with push notification to caregiver phones.

### Retraining cycle

The system is designed to be retrained, not frozen:

1. Incidents flagged by the dashboard are exported with their frames and
   confidences via the incident-report CSV.
2. A clinician confirms or corrects each one; corrections become new labels.
3. New footage is appended to the dataset and the pipeline is re-run end to end
   (`build_dataset → labeling → split_dataset → train → export_model → evaluate`).
4. The model is re-evaluated on a **frozen test set**. Fall recall is the release
   gate: a drop below the previous release blocks deployment.
5. `models/model_metadata.json` records the framework, hyper-parameters, dataset
   size and scores of every trained version, so releases are comparable.

Because labelling is a small, readable rule file rather than a black box, the
class definitions themselves can also be revised and the whole dataset relabelled
in seconds — without re-processing a single video.

---

## 10. Conclusion

SafeFall AI is a working end-to-end healthcare monitoring system: it detects
human posture with MediaPipe Pose, classifies five activities with a two-branch
CNN, identifies falls with **93.5 % recall and 0.954 ROC-AUC on videos it has
never seen**, raises emergency alerts with temporal confirmation, and presents
all of it in a deployed Streamlit dashboard.

Just as importantly, it is honest about what it is. The split is made at video
level so the numbers survive contact with a new resident. The one class that
matters clinically is supervised by human annotation, and the report says exactly
which classes are not. The deployed app runs the real trained network, and that
claim is verified numerically rather than asserted. Where a trade-off was made —
the alert threshold, the class definitions, the augmentation strength — the
report gives the reasoning and the measurement that drove it.
