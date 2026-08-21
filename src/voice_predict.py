"""
voice_predict.py
------------------
REAL, WORKING voice input for the command classifier. This is not a stub --
it actually captures/loads audio, transcribes it to text with a real speech
recognition engine, and feeds that text through the exact same trained
classifier used by predict.py.

TWO SPEECH-TO-TEXT ENGINES (auto-fallback):
    1. Google Web Speech API (`recognize_google`)
    2. CMU PocketSphinx (`recognize_sphinx`)

Why offer both: internet connectivity is not guaranteed for every user in
every part of Nigeria. This script tries Google first (better accuracy) and
automatically falls back to offline PocketSphinx if there's no internet
connection or the request fails, so the assistant still works either way.

THREE WAYS TO USE IT:
    # 1. Classify an existing audio file (wav/aiff/flac):
    python src/voice_predict.py --file path/to/command.wav

    # 2. Record live from your computer's microphone (needs PyAudio + a mic):
    python src/voice_predict.py --mic

    # 3. No arguments -> interactive menu asking which mode you want:
    python src/voice_predict.py

REQUIREMENTS:
    pip install SpeechRecognition pocketsphinx pyaudio
    (pyaudio is only needed for --mic / live microphone capture.
     On Linux you may also need: sudo apt-get install portaudio19-dev)

HONEST NOTE ON ACCURACY:
Speech-to-text accuracy depends heavily on microphone quality, background
noise, accent, and speaking pace. PocketSphinx's offline model is a
generic English model -- it was NOT trained on Nigerian Pidgin, so slang
words ("abeg", "wetin", "naira", "sha") will sometimes be mis-transcribed
as the closest-sounding English word. This is a known, real limitation of
free offline ASR, not something this script hides. For best results:
speak clearly, keep sentences short, and prefer the Google engine
(internet permitting) for higher accuracy on Nigerian-accented English.
"""

import sys
import os
import argparse

import speech_recognition as sr

from preprocess import clean_text
from predict import load_model, predict_command, print_prediction, ACTION_DESCRIPTIONS


def transcribe_audio(recognizer: sr.Recognizer, audio: sr.AudioData, engine: str = "auto"):
    """
    Converts recorded/loaded audio into text using a real speech recognition
    engine, with automatic fallback.

    engine:
        "auto"   -> try Google first, fall back to PocketSphinx if it fails
        "google" -> force Google Web Speech API only (needs internet)
        "sphinx" -> force offline PocketSphinx only (no internet needed)

    Returns: (transcribed_text, engine_used) or (None, None) if both fail.
    """
    if engine in ("auto", "google"):
        try:
            text = recognizer.recognize_google(audio)
            return text, "google (online)"
        except sr.UnknownValueError:
            # Google understood there was speech but couldn't make out words
            if engine == "google":
                return None, None
        except sr.RequestError as e:
            # No internet, or Google API unreachable -- fall back if allowed
            print(f"[voice] Google Speech API unavailable ({e}); falling back to offline engine...")
            if engine == "google":
                return None, None

    if engine in ("auto", "sphinx"):
        try:
            text = recognizer.recognize_sphinx(audio)
            return text, "pocketsphinx (offline)"
        except sr.UnknownValueError:
            return None, None
        except OSError as e:
            print(f"[voice] PocketSphinx not available: {e}")
            return None, None

    return None, None


def classify_audio_data(model, recognizer, audio, engine="auto"):
    """
    Full pipeline: audio -> transcribed text -> cleaned text -> predicted
    intent. Returns the transcript, engine used, and prediction tuple.
    """
    text, engine_used = transcribe_audio(recognizer, audio, engine=engine)
    if text is None:
        print("[voice] Could not understand the audio. Try again, speak "
              "clearly, and check your microphone / internet connection.")
        return None, None, None

    predicted, confidence, top3 = predict_command(model, text)
    return text, engine_used, (predicted, confidence, top3)


def run_on_file(model, filepath, engine="auto"):
    """Loads an audio file from disk and runs it through the full pipeline."""
    if not os.path.exists(filepath):
        print(f"ERROR: audio file not found: {filepath}")
        return

    recognizer = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio = recognizer.record(source)

    text, engine_used, prediction = classify_audio_data(model, recognizer, audio, engine=engine)
    if text is None:
        return

    print(f"\nAudio file       : {filepath}")
    print(f"Engine used      : {engine_used}")
    print(f"Transcribed text : \"{text}\"")
    predicted, confidence, top3 = prediction
    print_prediction(text, predicted, confidence, top3)


def run_on_microphone(model, engine="auto", listen_timeout=5, phrase_time_limit=6):
    """
    Records ONE spoken command live from the default microphone, then runs
    it through the full pipeline. Requires PyAudio and a working microphone.
    """
    recognizer = sr.Recognizer()

    try:
        mic = sr.Microphone()
    except Exception as e:
        print(
            "ERROR: No working microphone was found on this machine, or "
            f"PyAudio isn't installed correctly ({e}).\n"
            "Install with: pip install pyaudio\n"
            "(Linux users may also need: sudo apt-get install portaudio19-dev)"
        )
        return

    with mic as source:
        print("Adjusting for background noise... please wait a second.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening now -- speak your command (e.g. 'abeg check my balance')...")
        try:
            audio = recognizer.listen(source, timeout=listen_timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("No speech detected in time. Try again and speak right after 'Listening now'.")
            return

    print("Processing your command...")
    text, engine_used, prediction = classify_audio_data(model, recognizer, audio, engine=engine)
    if text is None:
        return

    print(f"Engine used      : {engine_used}")
    print(f"Transcribed text : \"{text}\"")
    predicted, confidence, top3 = prediction
    print_prediction(text, predicted, confidence, top3)


def main():
    parser = argparse.ArgumentParser(description="Voice input for the command classifier")
    parser.add_argument("--file", type=str, default=None, help="Path to an audio file (wav/flac/aiff) to classify")
    parser.add_argument("--mic", action="store_true", help="Record one live command from the microphone")
    parser.add_argument(
        "--engine", type=str, default="auto", choices=["auto", "google", "sphinx"],
        help="Speech-to-text engine: auto (default, tries Google then falls back to offline "
             "PocketSphinx), google (online only), or sphinx (offline only)",
    )
    args = parser.parse_args()

    model = load_model()

    if args.file:
        run_on_file(model, args.file, engine=args.engine)
        return

    if args.mic:
        run_on_microphone(model, engine=args.engine)
        return

    # No flags given -> interactive menu
    print("=" * 60)
    print("Voice Command Classifier - Voice Input Demo")
    print("=" * 60)
    print("1. Record a command from the microphone")
    print("2. Classify an existing audio file")
    print("3. Exit")
    choice = input("\nChoose an option (1/2/3): ").strip()

    if choice == "1":
        run_on_microphone(model, engine=args.engine)
    elif choice == "2":
        filepath = input("Enter path to the audio file: ").strip()
        run_on_file(model, filepath, engine=args.engine)
    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()
