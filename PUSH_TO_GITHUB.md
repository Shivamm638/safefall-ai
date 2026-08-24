# Submitting: GitHub → Streamlit Cloud → hand-in

Everything in this folder is already committed to a local git repository on the
`main` branch. Three steps remain, and the two GitHub ones need your account, so
they are yours to run.

---

## Step 1 — Create the empty repository (1 minute)

1. Go to <https://github.com/new> (sign in if needed).
2. **Repository name:** `safefall-ai`
3. **Visibility:** Public
4. **Do NOT tick** "Add a README", "Add .gitignore" or "Choose a license" —
   this folder already has them, and an initialised repo causes a push conflict.
5. Click **Create repository**.

GitHub then shows a page with commands. Ignore them and use Step 2 instead —
yours is already committed.

---

## Step 2 — Push (1 minute)

In a terminal **in this folder**, replacing `YOUR-USERNAME`:

```bash
git remote add origin https://github.com/YOUR-USERNAME/safefall-ai.git
git push -u origin main
```

A browser window or a "Sign in with your browser" prompt will appear the first
time — that is Git Credential Manager asking you to authorise the push. Approve
it. Nothing else needs your password.

If it says *remote origin already exists*, run this instead:

```bash
git remote set-url origin https://github.com/YOUR-USERNAME/safefall-ai.git
git push -u origin main
```

The upload is about 36 MB and takes under a minute.

---

## Step 3 — Deploy on Streamlit Cloud (~10 minutes, mostly waiting)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `YOUR-USERNAME/safefall-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** something readable, e.g. `safefall-ai`
4. Open **Advanced settings** and set **Python version to 3.11 or 3.12**.
   MediaPipe 0.10.21 publishes no wheels for 3.13 — picking 3.13 is the single
   most common way this build fails.
5. **Deploy**. The first build takes 5–10 minutes (it is compiling MediaPipe and
   OpenCV). Watch the log pane.

You get a public URL like `https://safefall-ai.streamlit.app`.

### Test the deployed app

Open it in a **fresh/incognito window** — that proves it does not depend on your
laptop, which is what the rubric asks for.

- [ ] Sidebar shows **● Model loaded and ready** and *NumPy ensemble of 3*
- [ ] **Upload & Analyse → Video →** bundled **Fall scenario** clip → emergency
      alert, timeline, key frames
- [ ] Same tab → **Normal activity** clip → no fall event, green status
- [ ] **Upload & Analyse → Image →** bundled fall sample → red alert
- [ ] **Analytics** populates after those runs
- [ ] **Model Performance** → all figures render
- [ ] **About** → pipeline diagram and model card

> **Note on the camera pages.** *Laptop Webcam* reads the camera on the machine
> running the app, so on Streamlit Cloud it will report no camera and offer
> browser snapshot capture instead — that is expected and handled. *CCTV / IP
> Camera* can only reach cameras that are publicly routable from the internet.
> Demonstrate both live pages from your **local** app in the video.

---

## Step 4 — Hand in

Put these in your submission:

1. **The live Streamlit URL** (top of the document — it is worth marks on its own)
2. **The GitHub repository URL**
3. **The evidence video** — record it with `docs/VIDEO_SCRIPT.md` open beside you
4. Optionally the screenshots you captured while testing

`docs/SUBMISSION_CHECKLIST.md` maps every rubric line to where it is evidenced.

---

## If something goes wrong

**Push rejected — "updates were rejected"**
You ticked one of the "initialise this repository" boxes. Either delete the repo
and recreate it empty, or run `git pull --rebase origin main` then push again.

**Build fails on `mediapipe`**
Python version. Settings → General → Python 3.11 or 3.12 → Reboot app.

**`ImportError: libGL.so.1`**
`packages.txt` did not get committed. Check with `git ls-files packages.txt`.

**"Trained model files were not found"**
`models/` did not get pushed. Check with `git ls-files models/`; if empty, run
`git add -f models/ && git commit -m "Add models" && git push`.

**App sleeps before the marker opens it**
Free Streamlit apps sleep after inactivity and take ~30 s to wake. Open your link
once shortly before submitting, and mention the first load may take a moment.
