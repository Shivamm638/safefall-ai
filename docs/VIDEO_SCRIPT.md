# Evidence Video — Recording Script

**Target length: 8–10 minutes.** Every rubric item is covered in order.
The narration below is written to be read aloud. Speak it in your own words if
you prefer — the important thing is that you *explain what each result means*,
not just click through screens. The rubric explicitly rewards explanation over
clicking.

---

## Before you hit record

**Checklist**

- [ ] `python -m streamlit run app.py` is running, or the Streamlit Cloud link is open
- [ ] The deployed link is open in a **fresh browser window** (prove it does not
      need your laptop)
- [ ] These files are open in tabs, ready to switch to:
      `results/fall_sequence.png`, `results/confusion_matrix.png`,
      `results/accuracy_graph.png`, `results/loss_graph.png`,
      `results/pose_estimation_grid.png`
- [ ] Sample media is ready — the app has bundled clips under
      **"No clip to hand? Try a bundled sample"**, so nothing to find
- [ ] Screen recorder set to capture the whole screen, microphone tested
- [ ] Close messaging apps and notifications

**Recording tips**

- Speak slightly slower than feels natural.
- Pause for a beat after each result appears, so the marker can read it.
- If you fluff a line, pause, and say it again cleanly — trim later or leave it.

---

## Part 1 — Project objective (about 45 seconds)

*Show: the dashboard home page.*

> "This is SafeFall AI, my FA-2 project: an AI-powered elderly fall detection
> and activity monitoring system.
>
> The problem it solves is not really the fall itself — it's what doctors call
> the *long lie*. When an older adult falls and nobody notices, every hour on the
> floor increases the risk of dehydration, pressure injury and pneumonia, and it
> is the strongest single predictor of whether they ever return to independent
> living. A camera-based monitor can't prevent the fall, but it can cut the time
> before someone finds them from hours to seconds — and unlike a wearable
> pendant, there's nothing for the resident to remember to put on.
>
> The system takes an image or a video from a room camera, works out the person's
> body posture, classifies what they're doing into one of five activities, and if
> it detects a fall it raises an immediate emergency alert."

---

## Part 2 — Dataset and classes (about 1 minute)

*Show: `docs/PROJECT_REPORT.md` section 2, or just talk over the dashboard.*

> "I used the Le2i Fall Detection dataset, carried forward from FA-1. It's 190
> videos recorded in six different rooms — coffee rooms, home settings, a lecture
> room and an office — at 320 by 240, 25 frames per second. That's about 76,000
> frames in total.
>
> 130 of those videos come with ground-truth annotation files that give the exact
> frame where each fall starts and ends. 96 videos contain a real fall.
>
> I ran MediaPipe pose estimation across all of them and kept every frame where
> it found a reliable skeleton — that's 21,470 frames, an 84.8 % detection rate.
>
> The system recognises five activities: **Fall Detected**, **Walking**,
> **Sitting**, **Standing**, and **Normal Activity** — which covers bending,
> reaching and crouching.
>
> One thing I want to be upfront about: the fall class comes from the human
> ground-truth annotations. But Le2i doesn't label walking, sitting or standing,
> so those four classes are derived from a transparent set of geometric rules —
> trunk angle, knee angle, stance width — which I've documented in full. So when
> you look at my results, the fall metrics are the properly supervised ones, and
> those are the ones the system should be judged on."

---

## Part 3 — Model choices and fall logic (about 1 minute 30)

*Show: the **About & Limitations** page in the dashboard — it has the pipeline
diagram.*

> "For pose estimation I used **MediaPipe Pose**, which gives 33 full-body
> landmarks — shoulders, elbows, wrists, hips, knees, ankles. I chose it for
> three reasons: it includes the lower-body detail that fall analysis actually
> needs, it runs in real time on a plain CPU with no GPU, and because it works
> from a skeleton rather than raw pixels it's inherently privacy-preserving —
> which matters a lot when you're filming residents continuously.
>
> For classification I built a **two-branch convolutional neural network**.
>
> The first branch treats the 33 landmarks as a one-dimensional signal and slides
> 1-D convolutions along the body's kinematic chain. That's the key idea: a fall
> is a *local geometric pattern* — the trunk rotating toward horizontal, the hips
> dropping, the legs collapsing — and convolutions share weights across body
> parts exactly the way an image CNN shares them across image patches.
>
> The second branch feeds in 25 hand-designed clinical features: trunk angle,
> knee angle, bounding-box aspect ratio, body height, stance width. These are the
> cues a physiotherapist would actually name, and they make the model's decisions
> explainable.
>
> The two branches are fused and go through dense layers to a five-way softmax.
> The whole model is about 128,000 parameters — small enough to run continuously
> on cheap hardware.
>
> The fall logic runs like this: frame goes in, MediaPipe extracts landmarks,
> features are computed, the CNN outputs an activity and a confidence. For a
> video, I smooth the fall probability over time and only latch an alert after
> five consecutive confirmed frames — that's what stops one bad frame causing a
> false alarm. I also merge alerts less than two seconds apart, because a
> caregiver should be told *a fall happened*, not handed four fragments of the
> same event."

---

## Part 4 — Pose estimation output (about 45 seconds)

*Show: `results/pose_estimation_grid.png`.*

> "Here's the pose estimation working across all five activity classes. You can
> see the 33 landmarks and the skeleton overlay on each one.
>
> Look at the difference between the standing frame and the fall frame — when the
> person is upright the skeleton is tall and narrow, and when they're on the floor
> it's wide and flat. That geometric difference is exactly what the trunk angle
> and the bounding-box aspect ratio features capture, and it's what the network
> learns from."

---

## Part 5 — Training and evaluation results (about 1 minute 30)

*Show: the **Model Performance** page in the dashboard.*

> "Now the results. The most important methodological point first: I split the
> data **70/15/15 at video level, not frame level**.
>
> That matters enormously. Neighbouring frames of the same video are almost
> identical, so if you split randomly by frame you end up testing on
> near-duplicates of your training data and you get a beautiful accuracy that
> means nothing. By splitting whole videos — 91 for training, 19 for validation,
> 20 for testing — the test set is genuinely unseen. It reports a lower number,
> and it's the only number that's worth anything.
>
> On those 20 unseen test videos — 1,975 frames — the model gets:
>
> - **Accuracy 83.7 %**
> - **Precision 82.1 %**, **Recall 79.0 %**, **F1-score 79.9 %** as macro averages
> - and weighted averages around 81 %."

*Scroll to the fall-class metrics row.*

> "But for a fall-detection system, the number that really matters is **recall on
> the fall class** — what fraction of real falls do we actually catch? Because a
> missed fall means a resident lying on the floor with nobody coming. A false
> alarm just costs a caregiver five seconds to dismiss.
>
> **Fall recall is 93.5 %** at the default decision, and **95.8 % at the alert
> threshold the system actually deploys** — so we catch more than nine out of ten
> falls. Fall precision is 77 %, and the **ROC-AUC is 0.954**, which tells you the
> model almost always ranks a real fall frame above a non-fall frame."

*Scroll to the threshold analysis plot.*

> "This plot is why the threshold isn't just left at 0.5. Notice that precision
> barely moves across the whole range — 77 to 78 % — while recall swings nine
> percentage points. So lowering the alert threshold is nearly free: I catch five
> percent more falls for almost no extra false alarms. I picked the threshold on
> the *validation* set using an explicit rule, so I'm not tuning on my test data."

---

## Part 6 — Accuracy and loss graphs (about 45 seconds)

*Show: `accuracy_graph.png` then `loss_graph.png` (both on the Model
Performance page).*

> "Here are the training curves. Training accuracy climbs to about 95 % while
> validation peaks at 89 % — the gap is normal overfitting, and early stopping
> restored the best validation weights rather than the final ones.
>
> The loss curves tell the same story: validation loss bottoms out early and then
> starts to rise, which is exactly the point where the model stops learning
> general posture and starts memorising specific training videos. That's the
> epoch the system keeps.
>
> I also fought that overfitting with data augmentation — I generate four extra
> versions of every training skeleton by mirroring it, rescaling it, shifting it,
> adding landmark noise, and simulating occlusion where furniture hides a limb."

---

## Part 7 — Confusion matrix (about 1 minute 15)

*Show: `results/confusion_matrix.png`, and the "Written analysis" expander.*

> "The confusion matrix. Let me read it properly rather than just showing it.
>
> The critical row is the fall row: **401 of 429 real fall frames were correctly
> detected**, and 54 were missed. Specificity is 92.9 %, so about 7 % of non-fall
> frames triggered an alert.
>
> Now, *where* do those false alarms come from? Almost entirely from Normal
> Activity — that's bending and crouching, which are literally the postures
> halfway between standing up and lying on the floor. And a bit from Sitting,
> because sitting on a chair and sitting on the floor after a fall look
> geometrically similar.
>
> What's much more important is what *doesn't* trigger alerts: only 3 standing
> frames and 2 walking frames out of nearly a thousand were misread as falls. An
> upright, mobile resident essentially never sets off a false alarm.
>
> The other main confusion is **Standing versus Walking**, and that one is
> genuinely unavoidable from a single image — the only difference between
> standing with your feet apart and being mid-stride is *motion*, and one still
> frame can't show motion. That's why my video pipeline adds a motion-refinement
> step that corrects those two classes once several frames are available.
>
> And here's the summary statistic I'd point to: **53 % of all the model's errors
> are swaps between two safe activities** that never change the alert a caregiver
> sees. So the clinically meaningful error rate is much lower than the raw
> accuracy figure suggests."

---

## Part 8 — Live dashboard: a normal prediction (about 1 minute)

*Go to the deployed Streamlit link. **Live Monitor → Image upload**. Open the
sample expander, pick the **Normal activity** sample, click Analyse.*

> "Now the live system. This is the deployed Streamlit app running in the cloud —
> I'll show the link at the end.
>
> I'll analyse a normal activity frame first."

*Wait for the result.*

> "On the left is the input frame, on the right is the pose estimation with all
> 33 landmarks drawn on. The system reports **Standing**, with high confidence,
> and a green safe status.
>
> Notice it doesn't just give a label — it explains itself: it reports the
> measured trunk angle and the body aspect ratio behind the decision. If a
> caregiver is going to trust an automated alert, they need to see the reasoning,
> not just a number.
>
> And below that, the full probability distribution across all five classes."

---

## Part 9 — Live dashboard: fall alert demonstration (about 1 minute 30)

*Go to **Video upload**, open the sample expander, choose a **Fall scenario**
clip, click Analyse.*

> "Now the important one — a real fall video from the held-out test set, footage
> the model has never seen."

*Wait for processing.*

> "Straight away you get the **emergency alert**: a pulsing red banner, an audible
> alarm, the time the fall happened, how long it lasted, the peak confidence, the
> severity, and a recommended action for the caregiver.
>
> Then the monitoring analytics: total activities detected, how many frames were
> classified as a fall, how many were normal activity, and the average confidence.
>
> This chart is the fall probability over time. You can see it sitting near zero
> while the person is walking around, then spiking sharply the moment they fall
> and *staying* high — because they're still on the floor. The dashed line is the
> alert threshold, and the shaded region is the confirmed fall event.
>
> Below that, the activity distribution for the clip and a timeline showing which
> activity was detected at each second.
>
> And here are the key frames with pose overlays — you can watch it go from
> walking, to the fall itself, to the person on the floor with the alert still
> latched.
>
> Everything can be exported as an incident report CSV, which is what you'd hand
> to a clinician or attach to a care record."

*Click into the **Monitoring Analytics** page.*

> "And the analytics page accumulates everything across the session — total
> activities, fall count, normal count, average confidence, the activity
> distribution, a resident safety index, and a full event log."

---

## Part 10 — Limitations and future improvements (about 1 minute)

*Show: the **About & Limitations** page.*

> "I want to be honest about the limitations, because a system like this only
> gets deployed if you're clear about where it fails.
>
> **Lighting** — pose detection drops off in poor light, and I only got a usable
> skeleton in 85 % of frames overall. An infrared camera would fix that, and
> night is exactly when falls happen most.
>
> **Occlusion** — if furniture hides the hips or legs, MediaPipe returns a partial
> skeleton. I chose to report 'no person detected' rather than guess, and I
> trained with simulated occlusion to harden it.
>
> **Camera angle** — I normalise by torso length so distance is handled well, but
> extreme overhead angles are still hard.
>
> **Similar postures** — sitting on a chair versus sitting on the floor after a
> fall is a genuinely hard distinction, and you can see it in my confusion matrix.
>
> **And the dataset itself** — these are staged falls performed by younger actors.
> Real elderly falls are slower and more varied, so I'd treat my numbers as an
> upper bound until it's validated on genuine care-home footage.
>
> For future improvements: a temporal model over a window of frames to catch slow
> 'slump' falls, real elderly footage, low-light and infrared data, a short 'are
> you OK?' confirmation before escalating to reduce false alerts, and real-time
> CCTV support with push notifications to caregivers' phones.
>
> On retraining: the system is designed to be retrained, not frozen. Confirmed
> incidents get exported, a clinician corrects the labels, they're appended to the
> dataset, and the whole pipeline re-runs. The model is re-evaluated on a frozen
> test set, and **fall recall is the release gate** — if it drops below the
> previous version, that release doesn't ship."

---

## Part 11 — The deployed link (about 30 seconds)

*Show the browser address bar clearly. Open the link in a fresh window if you
haven't already.*

> "And finally — this is the live deployed link on Streamlit Cloud. It's running
> in a fresh browser session, not from my local machine. The model, the pose
> estimation and the dashboard are all running in the cloud.
>
> One engineering detail I'm proud of: TensorFlow is about 600 megabytes and
> Streamlit's free tier only gives you a gigabyte of RAM, which TensorFlow plus
> MediaPipe plus OpenCV won't reliably fit inside. So instead of shrinking the
> model, I exported the trained weights and re-implemented the identical network
> in pure NumPy. And I didn't just assume it worked — I wrote a parity check that
> compares both engines across the entire test set. They agree to within one times
> ten to the minus six, with 100 % identical predictions, and the deployment
> script refuses to ship if they ever disagree.
>
> That's SafeFall AI. Thank you for watching."

---

## Coverage check

Tick these off after your first take — each is a rubric line.

| # | Rubric item | Covered in |
|---|---|---|
| 1 | Project objective | Part 1 |
| 2 | Dataset and classes | Part 2 |
| 3 | Model choices and fall logic | Part 3 |
| 4 | Pose-estimation output | Part 4 |
| 5 | Training / evaluation results | Part 5 |
| 6 | Accuracy, precision, recall, F1 | Part 5 |
| 7 | Confusion matrix + graphs | Parts 6 and 7 |
| 8 | Streamlit dashboard | Parts 8 and 9 |
| 9 | Normal prediction | Part 8 |
| 10 | Fall alert demonstration | Part 9 |
| 11 | Limitations / future improvements | Part 10 |
| 12 | Working deployed link | Part 11 |

**The single biggest scoring tip:** the rubric says *"Explain instead of only
clicking."* After every result you show, add one sentence saying what it means
for an elderly resident. That is the difference between a good video and a 5/5
video.
