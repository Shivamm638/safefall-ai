# Deploying SafeFall AI to Streamlit Community Cloud

The assignment requires a **hosted Streamlit link that works without your
laptop**. This guide takes you from this folder to a public URL.

Budget about 15 minutes, most of it waiting for the first build.

---

## What gets deployed

Everything the app needs is already in this folder and is small enough for
GitHub:

| Item | Size | Why it is needed |
|---|---|---|
| `app.py` | 35 KB | the dashboard |
| `src/` | 120 KB | pipeline code |
| `models/fall_detection_cnn_weights.npz` | **469 KB** | the trained network |
| `models/feature_scaler.json` | 12 KB | training-time normalisation |
| `models/model_metadata.json` | 2 KB | model card shown in the app |
| `models/fall_detection_cnn.keras` | 1.6 MB | the Keras model (kept for reference) |
| `results/*.png` | ~3 MB | the plots shown on the Model Performance page |
| `data/samples/` | ~1.5 MB | bundled demo clips and stills |
| `requirements.txt`, `packages.txt`, `.streamlit/config.toml` | tiny | build configuration |

**Total: under 10 MB.** The 17 GB of raw video and the intermediate feature
tables are excluded by `.gitignore` — they are only needed to retrain.

---

## Step 0 — Run the pre-flight check

```bash
python -m src.verify_deployment
```

This hides every training-only package and then drives the entire app the way
Streamlit Cloud will — model load, pose estimation, image prediction, video
analysis, alert logic — and confirms every file the app needs is present. It
finishes with `PASSED` if you are safe to deploy.

Do this before pushing. It is much faster to fix a missing dependency here than
to wait 8 minutes for a cloud build to fail.

---

## Step 1 — Put the project on GitHub

If you do not have Git installed, get it from <https://git-scm.com/downloads>.

Open a terminal in this folder (`FA 2 Output`) and run:

```bash
git init
git add .
git commit -m "SafeFall AI - FA-2 elderly fall detection system"
```

Check that the model files really were included — this is the single most common
deployment failure:

```bash
git ls-files models results/metrics_summary.json data/samples
```

You should see `fall_detection_cnn_weights.npz`, `feature_scaler.json`,
`model_metadata.json`, `metrics_summary.json` and the sample media listed. If
`models/` is empty, force-add it:

```bash
git add -f models/ results/ data/samples/
git commit -m "Add trained model, results and demo media"
```

Now create an **empty public repository** on GitHub called `safefall-ai`
(no README, no .gitignore — this folder already has them), then:

```bash
git remote add origin https://github.com/YOUR-USERNAME/safefall-ai.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with your GitHub account.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `YOUR-USERNAME/safefall-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose something readable, e.g. `safefall-ai`
4. Leave the **Python version** alone — the default is fine. `requirements.txt`
   installs a MediaPipe build that publishes wheels for every interpreter
   Streamlit Cloud offers, and `src/pose_backend.py` adapts to whichever
   MediaPipe API that build exposes, so no version needs choosing by hand.
5. Click **Deploy**.

The first build takes 8–12 minutes — it is installing MediaPipe, OpenCV and the
WebRTC stack.
Watch the log pane; when it finishes you get a public URL of the form
`https://safefall-ai.streamlit.app`.

---

## Step 3 — Test the deployed app properly

The rubric asks you to test the deployed version, not just to see it load. Open
the URL in a **fresh browser window** — ideally a private/incognito one, or on
your phone — so you have proved it does not depend on your machine.

Then check every item:

- [ ] The page loads and the sidebar shows **"Model loaded and ready"**
- [ ] The sidebar reports the inference engine as **NumPy (exported Keras weights)**
- [ ] Sidebar shows test accuracy and fall recall
- [ ] **Live Monitor → Image upload** → analyse a bundled sample → pose overlay,
      prediction and confidence all appear
- [ ] Analyse a **fall** sample image → the red emergency alert fires
- [ ] **Video upload** → analyse a bundled fall clip → fall event detected, the
      probability timeline renders, key frames appear
- [ ] Analyse the **normal activity** clip → no fall event, green safe status
- [ ] **Monitoring Analytics** → counters, distribution charts and event log populate
- [ ] **Model Performance** → all six plots render
- [ ] **About & Limitations** → pipeline diagram and model card render
- [ ] Download an incident report CSV

Take screenshots as you go — the rubric asks for the deployed dashboard, a
normal prediction, a fall prediction with alert, pose output, metrics and the
confusion matrix. Save them into `screenshots/`.

---

## Troubleshooting

**Build fails on `mediapipe` — "no wheels with a matching Python ABI tag"**
This is the problem the project used to have, and it is worth understanding
because the fix shapes several files. MediaPipe split into two incompatible
eras and no single release spans them:

| MediaPipe | 3.13 / 3.14 wheels | `mp.solutions` API |
|---|---|---|
| ≤ 0.10.21 | no | **yes** — what the pipeline was written against |
| ≥ 0.10.30 | yes | **removed** |

Pinning the old release fails to *build* on a modern interpreter; pinning a new
one fails at *import*. Rather than depend on someone remembering to set a Python
version in a web form, `src/pose_backend.py` detects which API is present and
drives the same BlazePose network either way. `requirements.txt` therefore pins
`mediapipe==0.10.33`, whose wheel is tagged `py3-none-manylinux_2_28_x86_64` —
no interpreter constraint at all — so it installs on 3.12 and 3.14 alike.

If you still hit this, `requirements.txt` has been edited back to a `0.10.21`
pin. Restore `mediapipe==0.10.33`.

**Build fails with `ResolutionImpossible` mentioning `streamlit-webrtc`**
`streamlit-webrtc` 0.77 requires `streamlit >= 1.51`. If `requirements.txt`
pins an older Streamlit the two cannot co-exist and pip refuses the whole file,
so *nothing* installs. Keep `streamlit==1.62.0`.

Note that pinning Streamlit also constrains protobuf: modern Streamlit requires
`protobuf >= 5.26.1`, while legacy MediaPipe required `protobuf < 5`. Those two
cannot both be satisfied — another reason the legacy MediaPipe branch is gone.

**Build fails in the apt step with "held broken packages"**
`packages.txt` is asking for a package name that does not exist on Streamlit
Cloud's Debian **trixie** image. `libglib2.0-0` is the usual culprit: it was
renamed `libglib2.0-0t64` in the 64-bit `time_t` transition, so the old name
only matches the stale bullseye repo still listed in the image's sources and
drags in `libffi7`/`libpcre3`, which trixie does not ship. Use the `t64` name.

**`ImportError` from `cv2/__init__.py`, in `bootstrap()`**
The Python side of OpenCV imported fine and then failed to load its *native*
module, which means a shared library it links against is absent. On Streamlit
Cloud the message is redacted in the browser — open **Manage app** to see which
`lib*.so` it names.

`pip install opencv-contrib-python` does **not** install these. The wheel
vendors 42 libraries but deliberately leaves the graphical and system ones to
the OS, and they are what `packages.txt` exists to supply:

| OpenCV needs | Debian trixie package |
|---|---|
| `libGL.so.1` | `libgl1` |
| `libglib-2.0.so.0`, `libgthread-2.0.so.0` | `libglib2.0-0t64` |
| `libSM.so.6` | `libsm6` |
| `libICE.so.6` | `libice6` |
| `libXext.so.6` | `libxext6` |
| `libX11.so.6` | `libx11-6` |
| `libxcb.so.1` | `libxcb1` |
| `libz.so.1` | `zlib1g` |

This list is not folklore — it is the `DT_NEEDED` table of every `.so` in the
wheel, minus the ones the wheel ships itself. It is identical for OpenCV 4.14
and 5.0, so changing OpenCV version never fixes it.

The trap is that **every `import cv2` in this project is lazy**, deep inside a
function that only runs once you analyse something. A host missing these
libraries therefore starts up, renders, and looks perfectly healthy until the
first frame is processed. If you change `packages.txt`, prove it by actually
running an analysis on the deployed app, not by watching it load.

Keep `packages.txt` to bare package names, one per line, with no comments —
anything else is passed to `apt-get` as though it were a package.

**"Trained model files were not found"**
`models/` was not pushed — almost always because a global gitignore excluded it.
Fix with `git add -f models/ && git commit && git push`.

**The app runs out of memory**
Check that `requirements.txt` does **not** contain `tensorflow`. The deployed app
is designed to run the NumPy engine precisely so TensorFlow is never installed.
If you added it back, remove it and reboot.

**Video analysis is slow or times out**
Reduce `MAX_VIDEO_FRAMES` in `src/config.py` (900 by default) or increase
`VIDEO_FRAME_STRIDE`. The bundled samples are deliberately short for this reason.

**App has gone to sleep when the marker opens it**
Free Streamlit apps sleep after a period of inactivity and take ~30 seconds to
wake. Open your link once shortly before submitting, and mention in your
submission that the first load may take a few seconds.

---

## After deploying

1. Put the live URL at the top of your submission document.
2. Add it to `README.md` so the link travels with the code.
3. Record the evidence video with `docs/VIDEO_SCRIPT.md` open beside you, and
   show the deployed link on screen at the end (Part 11 of the script).
