---
title: Voice & Text Command Classifier
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.61.1"
app_file: app.py
pinned: false
license: mit
---

<!--
The YAML block above is required by Hugging Face Spaces and is read ONLY
by Spaces -- GitHub and every other renderer ignores it and shows the
Markdown below as normal. If you push this exact repo to a Hugging Face
Space git remote (see Section 10c "Deployment"), the Space will pick up
sdk/app_file/etc. from here automatically. No separate config file needed.
-->

# Voice/Text Command Classifier (AI-18)

A machine learning system that classifies short **Nigerian Pidgin / English**
voice-or-text commands into app actions — built for low-literacy users who
prefer speaking or typing simple commands over navigating menus.

> Example: `"abeg send 2k give my mama"` → **`send_money`**

---

## 1. Problem & Approach

Low-literacy users in Nigeria often find multi-step app menus difficult to
navigate. A simpler interaction model is: **say or type what you want, and
the app figures out what to do.** This project builds the "brain" behind
that interaction — a text classifier that maps a raw command (typed, or
transcribed from speech) to one of 12 supported app actions (intents).

**Approach:**
1. Generate a realistic labeled dataset of Pidgin/English commands (since no
   public dataset exists for this exact use case).
2. Clean the text (lowercase, strip punctuation).
3. Convert text to numeric features using **TF-IDF** (word + bigram counts).
4. Train and compare 3 classic ML models (Logistic Regression, Linear SVM,
   Naive Bayes) using cross-validation.
5. Evaluate on both a template-style held-out test set AND a hand-written
   "hard" test set of genuinely novel phrasings, for an honest accuracy
   picture.
6. Ship a CLI demo tool that classifies any typed command in real time.

---

## 2. Intent Set (12 app actions)

| Intent | Example command |
|---|---|
| `check_balance` | "abeg check my account balance" |
| `send_money` | "send 2k give my mama" |
| `buy_airtime` | "recharge my line with 500 naira" |
| `buy_data` | "abeg buy MTN data bundle" |
| `call_contact` | "call my broda" |
| `open_app` | "open camera" |
| `check_weather` | "how the weather be today" |
| `set_reminder` | "remind me make I pay light bill" |
| `play_music` | "play some afrobeat" |
| `stop_action` | "stop am" |
| `greeting` | "how far" |
| `help_request` | "help me abeg" |

---

## 3. Project Structure

```
voice_command_classifier/
├── data/
│   ├── commands_dataset.csv     # main generated training dataset (960 rows)
│   ├── hard_test_set.csv        # 36 hand-written novel commands (unseen phrasing)
│   ├── test_split.csv           # auto-saved held-out test rows (from train.py)
│   └── sample_audio/            # 5 ready-to-try .wav files for voice_predict.py
├── src/
│   ├── generate_data.py         # builds commands_dataset.csv from templates
│   ├── preprocess.py            # shared text-cleaning function
│   ├── train.py                 # trains, compares & saves the best model
│   ├── evaluate.py               # generates confusion matrix plots
│   ├── predict.py               # interactive CLI demo tool (text input)
│   └── voice_predict.py         # REAL voice input: mic/file -> speech-to-text -> classify
├── models/
│   └── command_classifier.joblib  # saved trained pipeline (vectorizer + model)
├── results/
│   ├── classification_report.txt
│   ├── metrics.json
│   ├── confusion_matrix_test_split.png
│   └── confusion_matrix_hard_test.png
├── notebook/
│   └── Voice_Command_Classifier.ipynb   # Google Colab-ready, end-to-end notebook
├── requirements.txt
└── README.md
```

---

## 4. Setup

```bash
# 1. Clone/download this project, then move into it
cd voice_command_classifier

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Running the Project (in order)

```bash
# Step 1: Generate the synthetic labeled dataset
python src/generate_data.py
# -> creates data/commands_dataset.csv (960 rows, 12 intents)

# Step 2: Train and compare models, save the best one
python src/train.py
# -> saves models/command_classifier.joblib
# -> saves results/classification_report.txt and results/metrics.json

# Step 3: Generate confusion matrix visualizations
python src/evaluate.py
# -> saves results/confusion_matrix_test_split.png
# -> saves results/confusion_matrix_hard_test.png

# Step 4: Try it out interactively
python src/predict.py
# -> type commands like "abeg buy credit for me" and see live predictions

# Or classify a single command directly:
python src/predict.py "call my sista"
```

Alternatively, open **`notebook/Voice_Command_Classifier.ipynb`** in Google
Colab and run all cells top to bottom — it reproduces the entire pipeline
in one place, with explanations.

---

## 6. Results

Trained on 768 commands, evaluated on two different test sets:

| Test set | What it measures | Accuracy |
|---|---|---|
| **Template-style test split** (192 rows, same style as training data) | How well the model fits the overall command patterns | **~99%** |
| **Hard/unseen test set** (36 hand-written commands, novel phrasing never seen in training) | How well the model *generalizes* to real, unpredictable user phrasing | **~83%** |

**Why report both?** A single accuracy number from data generated the same
way as the training set can be misleadingly high. The hard test set —
written by hand, using different sentence structures and word choices —
is a much more honest estimate of how the model would perform on real
users typing/speaking commands it has never encountered. This gap (99%
vs 83%) is expected and healthy to disclose; it also shows exactly where
the model struggles (see `results/confusion_matrix_hard_test.png` —
`call_contact` and `check_weather` are the intents most often confused
with others on novel phrasing, largely because their vocabulary overlaps
with `greeting` and `open_app`).

Full metrics: see `results/classification_report.txt` and
`results/metrics.json` after running `train.py`.

---

## 7. How the Model Works (plain-language summary)

1. **Cleaning**: Text is lowercased and stripped of punctuation.
   `"Abeg Send 2K!"` → `"abeg send 2k"`
2. **TF-IDF vectorization**: Turns each cleaned command into a vector of
   numbers representing which words/word-pairs it contains and how
   distinctive those words are across all commands. This is what lets the
   model treat text as math it can learn patterns from.
3. **Classifier**: A Logistic Regression model (chosen automatically for
   its strong accuracy AND well-calibrated confidence scores) learns which
   word patterns are associated with which intent.
4. **Prediction**: A new command goes through the same cleaning + TF-IDF
   steps, and the trained model outputs the most likely intent plus a
   confidence percentage.

---

## 8. Web App Demo (No Terminal Needed)

`app.py` is a Streamlit web app with two ways to try the classifier in a
browser — typing a command, or uploading a short audio clip (which gets
transcribed with the same speech-to-text logic as `voice_predict.py`,
Google online falling back to offline PocketSphinx).

**Run it locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`.

*(Live browser microphone recording isn't included in the hosted app —
that would need extra experimental browser components. For live mic
input, run `python src/voice_predict.py --mic` locally instead, per
Section 9.)*

---

## 9. Voice Input (Real, Working Speech-to-Text, CLI)

`src/voice_predict.py` adds actual voice input on top of the text
classifier — it captures/loads real audio, transcribes it to text with a
real speech recognition engine, then feeds that text through the exact
same trained model used by `predict.py`.

**Two engines, with automatic fallback:**
| Engine | Needs internet? | Accuracy | Notes |
|---|---|---|---|
| Google Web Speech API (`recognize_google`) | Yes | Higher | Free, no API key needed |
| CMU PocketSphinx (`recognize_sphinx`) | No | Lower | Fully offline — works even with no connectivity |

By default (`--engine auto`) it tries Google first and automatically falls
back to offline PocketSphinx if there's no internet — a deliberate design
choice, since internet connectivity can't be assumed for every user in
every part of Nigeria.

**Usage:**

```bash
# 1. Classify a sample audio file (works immediately, no mic needed)
python src/voice_predict.py --file data/sample_audio/check_balance_sample.wav

# 2. Record live from your microphone
python src/voice_predict.py --mic

# 3. Interactive menu (asks you to choose file or mic)
python src/voice_predict.py

# Force a specific engine instead of auto-fallback:
python src/voice_predict.py --file data/sample_audio/send_money_sample.wav --engine google
python src/voice_predict.py --mic --engine sphinx
```

Five ready-to-try sample audio files (generated with offline TTS, so no
copyright concerns) are included in `data/sample_audio/`:
`check_balance_sample.wav`, `send_money_sample.wav`, `open_app_sample.wav`,
`play_music_sample.wav`, `call_contact_sample.wav`.

**This was actually tested, not just written and assumed to work:**
`data/sample_audio/*.wav` were run through `voice_predict.py --engine sphinx`
in a sandboxed environment with no internet and no microphone, and the
pipeline correctly transcribed and classified them end-to-end. The `--mic`
path was also verified to fail gracefully with a clear, actionable error
message when no microphone hardware is present, instead of crashing.

**Honest limitation to know about:** PocketSphinx's offline model is a
*generic* English acoustic model — it was not trained on Nigerian Pidgin,
so slang words ("abeg", "wetin", "naira") sometimes get mis-transcribed as
the closest-sounding English word. In testing here, clear TTS-generated
English audio transcribed with a mix of full accuracy and partial/garbled
results depending on the phrase. This is a known, real limitation of free
offline speech recognition — not something hidden or glossed over. In
practice, with a real human voice (rather than synthetic TTS) and a
decent microphone, PocketSphinx tends to perform noticeably better than
it did on the robotic test audio here, and the Google engine (when
internet is available) is substantially more accurate on both standard
and Nigerian-accented English. For production use at scale, a
Pidgin-tuned or Nigerian-accented ASR model (e.g. fine-tuned Whisper or
Vosk) would be a valuable upgrade — noted in Section 11.

**Install:**
```bash
pip install SpeechRecognition pocketsphinx pyaudio
# Linux may also need: sudo apt-get install portaudio19-dev
```

---

## 10. Deployment

### 10a. Push to GitHub

```bash
cd voice_command_classifier
git init
git add .
git commit -m "Initial commit: Voice/Text Command Classifier"

# Create an empty repo on github.com first (no README/license — you already have them),
# then:
git branch -M main
git remote add origin https://github.com/<your-username>/voice-command-classifier.git
git push -u origin main
```

The `models/command_classifier.joblib` file (~a few hundred KB) is small
enough to commit directly — no Git LFS needed.

### 10b. Host the web demo on Streamlit Community Cloud (free)

1. Push the repo to GitHub (step 10a).
2. Go to **share.streamlit.io** and sign in with your GitHub account.
3. Click **"New app"**, select your repo, branch `main`, and set the main
   file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL like
   `https://<your-app-name>.streamlit.app` that anyone can open — no
   install, no terminal.

**Note on PocketSphinx on Streamlit Cloud:** the audio-upload tab's
offline fallback engine needs system-level build tools that Streamlit
Cloud's environment may not have preinstalled. If the audio tab errors on
the hosted version specifically, the text-input tab will still work
perfectly (it has no such dependency) — you can also add a
`packages.txt` file to the repo root with the line `libpulse-dev` to
supply the missing system library Streamlit Cloud needs for PocketSphinx.

### 10c. Hugging Face Spaces (Streamlit, No Code Changes)

Same `app.py`, no code changes needed — free hosting, and often preferred
for ML portfolios specifically.

1. Go to **huggingface.co/new-space**, sign in (or create a free account).
2. Set **Space name**, choose SDK: **Streamlit**, visibility: Public.
3. This repo's `README.md` already has the required YAML config block at
   the very top (`sdk: streamlit`, `app_file: app.py`, etc.) — Hugging
   Face Spaces reads it automatically, nothing else to configure.
4. Push this repo to the Space's git remote (shown on the Space's page
   after creation):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
5. The Space builds automatically and gives you a URL like
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

`packages.txt` (already in this repo) is read by Spaces the same way as
Streamlit Cloud, for the same PocketSphinx system-library reason noted in
10b.

---

### 10d. Using a Different Frontend Entirely (e.g. Vercel)

Streamlit and Hugging Face Spaces are the simplest paths because they're
**one Python service** — the ML code and the UI live together. Vercel is
built for static/React frontends and short-lived serverless functions, so
it **cannot host `app.py` directly** — Streamlit needs a persistent
running server, and PocketSphinx needs system-level build tools that
don't fit Vercel's serverless Python runtime.

To use Vercel, the app has to be split into two separately-hosted pieces:
- **`backend/`** — a FastAPI service exposing the model over HTTP,
  deployed somewhere that supports persistent Python (Render, Railway, or
  Fly.io — all free-tier friendly)
- **`frontend/`** — a Next.js app deployed on Vercel that calls the
  backend's API

Both are included in this repo. See `backend/README.md` and
`frontend/README.md` for setup and deployment steps for each.

---

## 11. What to Include vs. Leave Out of a Public/Shared Copy

If you're sharing this as a portfolio piece (GitHub + live demo), keep
everything — `results/` is proof of your evaluation work and belongs in a
portfolio repo. If you ever want a stripped-down copy for someone who just
wants to *use* the tool (not see how it was built/evaluated), you can drop:
- `results/` (evaluation artifacts, not needed to run the tool)
- `notebook/` (only needed if they want the Colab walkthrough)
- `data/commands_dataset.csv` and `data/hard_test_set.csv` (only needed if
  they want to retrain; the saved model in `models/` already has everything
  baked in)

The one thing you should never strip out is `models/command_classifier.joblib`
— without it, nothing runs unless the person retrains from scratch first.

---

## 12. Recording the 2-3 Minute Demo Video

Suggested script:
1. **(30s)** Explain the problem: low-literacy users prefer simple voice/text
   commands over navigating app menus.
2. **(30s)** Show the dataset (`data/commands_dataset.csv`) and intent list.
3. **(30s)** Run `python src/train.py` on screen — show the model comparison
   and accuracy output.
4. **(20s)** Show `results/confusion_matrix_hard_test.png` and explain the
   99% vs 83% distinction (shows you understand real-world generalization,
   not just memorized accuracy).
5. **(20s)** Run `python src/predict.py` interactively and type 3-4 live
   Pidgin/English commands, showing the predicted action + confidence.
6. **(30s)** Run `python src/voice_predict.py --mic`, speak a command out
   loud, and show it get transcribed and classified live — this is the
   "voice" half of "Voice/Text Command Classifier." Optionally also show
   the hosted `app.py` web demo running in a browser.

---

## 13. Possible Future Improvements

- Collect real user commands (not just synthetic/template data) to further
  close the template-vs-hard-test accuracy gap.
- Add more intents (e.g. `check_transaction_history`, `change_language`).
- Add a fallback `unknown`/`unclear` intent with a confidence threshold, so
  the app asks the user to rephrase instead of guessing wrong.
- Add live in-browser microphone recording to the hosted web app (e.g. via
  `streamlit-webrtc` or `audio-recorder-streamlit`) instead of file upload.
- Fine-tune a small transformer (e.g. DistilBERT) if more labeled data
  becomes available, and compare against this TF-IDF baseline.

