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
4. Open **Advanced settings** and set **Python version to 3.11** or **3.12**.
   MediaPipe 0.10.21 does not publish wheels for 3.13, and picking a version
   without wheels is the second most common deployment failure.
5. Click **Deploy**.

The first build takes 5–10 minutes — it is installing MediaPipe and OpenCV.
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

**Build fails on `mediapipe`**
The Python version is wrong. Go to **Settings → General**, set Python to 3.11 or
3.12, and reboot the app. MediaPipe has no wheels for 3.13.

**`ImportError: libGL.so.1` or `libgthread`**
`packages.txt` is missing or was not committed. It must contain `libgl1` and
`libglib2.0-0`. Confirm with `git ls-files packages.txt`.

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
