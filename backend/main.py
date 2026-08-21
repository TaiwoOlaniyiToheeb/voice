"""
main.py
--------
FastAPI backend for the Voice/Text Command Classifier.

This exposes the exact same trained model and classification logic used
by predict.py / voice_predict.py / app.py as an HTTP API, so a separately
hosted frontend (e.g. a Next.js app on Vercel) can call it over the network.

WHY A SEPARATE BACKEND?
Vercel hosts static sites and short-lived serverless functions -- it
can't run a persistent Python process with scikit-learn/PocketSphinx
loaded in memory. This backend is meant to be deployed somewhere that
DOES support a persistent server (Render, Railway, Fly.io), while the
frontend on Vercel just makes HTTP requests to it.

ENDPOINTS:
    GET  /                  -> health check
    GET  /intents            -> list of supported intents + descriptions
    POST /predict/text       -> {"text": "..."} -> prediction JSON
    POST /predict/audio      -> multipart audio file upload -> transcribe + predict

Run locally with:
    uvicorn main:app --reload --port 8000

Test locally:
    curl -X POST http://localhost:8000/predict/text \\
         -H "Content-Type: application/json" \\
         -d '{"text": "abeg send 2k give my mama"}'
"""

import os
import tempfile

import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import clean_text

# ----------------------------------------------------------------
# Model loading (once, at startup -- not per-request)
# ----------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "command_classifier.joblib")
model = joblib.load(MODEL_PATH)

ACTION_DESCRIPTIONS = {
    "check_balance": "Show account balance screen",
    "send_money": "Open send-money / transfer flow",
    "buy_airtime": "Open buy-airtime flow",
    "buy_data": "Open buy-data-bundle flow",
    "call_contact": "Open phone dialer / place a call",
    "open_app": "Launch the requested app",
    "check_weather": "Show weather forecast",
    "set_reminder": "Create a new reminder/alarm",
    "play_music": "Open music player and play",
    "stop_action": "Stop/cancel the current action",
    "greeting": "Reply with a friendly greeting",
    "help_request": "Show help / assistant guidance",
}

# ----------------------------------------------------------------
# FastAPI app + CORS
# ----------------------------------------------------------------
app = FastAPI(
    title="Voice/Text Command Classifier API",
    description="Classifies Nigerian Pidgin/English commands into app actions.",
    version="1.0.0",
)

# CORS: allows the Vercel frontend (and local dev) to call this API from
# a browser. In production, replace "*" with your actual Vercel domain
# (e.g. "https://your-app.vercel.app") for tighter security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voicetextcommand.vercel.app/"],  # tighten this to your Vercel domain before going live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


def classify(raw_text: str):
    """Shared classification logic used by both endpoints."""
    cleaned = clean_text(raw_text)
    predicted = model.predict([cleaned])[0]
    probs = model.predict_proba([cleaned])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)
    confidence = float(dict(ranked)[predicted])
    top5 = [{"intent": intent, "confidence": float(p)} for intent, p in ranked[:5]]
    return {
        "command": raw_text,
        "predicted_intent": predicted,
        "description": ACTION_DESCRIPTIONS.get(predicted, ""),
        "confidence": confidence,
        "top_5": top5,
    }


@app.get("/")
def health_check():
    """Simple health check -- used by Render/Railway/Fly.io to confirm the
    service is alive, and by you to sanity-check the deployment worked."""
    return {"status": "ok", "service": "voice-command-classifier-api"}


@app.get("/intents")
def list_intents():
    """Returns the full supported intent set + human-readable descriptions."""
    return {"intents": ACTION_DESCRIPTIONS}


@app.post("/predict/text")
def predict_text(request: TextRequest):
    """Classifies a typed command."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="`text` must not be empty.")
    return classify(request.text)


@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """
    Accepts an uploaded audio file (wav/flac/aiff), transcribes it with
    the same Google-then-PocketSphinx fallback logic used throughout the
    rest of this project, then classifies the transcribed text.
    """
    import speech_recognition as sr

    allowed_extensions = (".wav", ".flac", ".aiff", ".aif")
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use one of: {allowed_extensions}",
        )

    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)

        text, engine_used = None, None
        try:
            text = recognizer.recognize_google(audio)
            engine_used = "google (online)"
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass  # fall through to offline engine

        if text is None:
            try:
                text = recognizer.recognize_sphinx(audio)
                engine_used = "pocketsphinx (offline)"
            except (sr.UnknownValueError, OSError):
                pass

        if text is None:
            raise HTTPException(
                status_code=422,
                detail="Could not understand the audio. Try a clearer recording.",
            )

        result = classify(text)
        result["transcription_engine"] = engine_used
        return result
    finally:
        os.remove(tmp_path)
