"""
predict.py
-----------
Interactive command-line demo of the trained classifier.

This is the script you'd run on camera for the 2-3 minute demo video:
it takes a typed command (standing in for a voice-transcribed command)
and prints the predicted app action + how confident the model is.

USAGE:
    # Interactive mode - keeps asking for commands until you type 'exit'
    python src/predict.py

    # Single command mode - classify one command and exit
    python src/predict.py "abeg send 2k give my mama"

NOTE ON VOICE INPUT:
The brief says "Voice/Text Command Classifier". The classification core
(TF-IDF + ML model) works on TEXT. For actual VOICE input, you first need
speech-to-text. This script accepts typed text directly (fastest to demo
reliably without a microphone/internet dependency in a recorded video).
See the README's "Optional: real microphone voice input" section for how
to plug in the `speech_recognition` library so spoken commands are
transcribed to text and then passed through this exact same classifier.
"""

import sys
import os
import joblib

from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "command_classifier.joblib")

# Human-friendly descriptions shown alongside the predicted intent, so a
# non-technical demo viewer immediately understands what action would fire.
ACTION_DESCRIPTIONS = {
    "check_balance": "-> Show account balance screen",
    "send_money": "-> Open send-money / transfer flow",
    "buy_airtime": "-> Open buy-airtime flow",
    "buy_data": "-> Open buy-data-bundle flow",
    "call_contact": "-> Open phone dialer / place a call",
    "open_app": "-> Launch the requested app",
    "check_weather": "-> Show weather forecast",
    "set_reminder": "-> Create a new reminder/alarm",
    "play_music": "-> Open music player and play",
    "stop_action": "-> Stop/cancel the current action",
    "greeting": "-> Reply with a friendly greeting",
    "help_request": "-> Show help / assistant guidance",
}


def load_model():
    """Loads the trained (vectorizer + classifier) pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        print(
            "ERROR: No trained model found at "
            f"{MODEL_PATH}\nRun 'python src/train.py' first to train and save a model."
        )
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def predict_command(model, raw_text: str):
    """
    Cleans the raw input text the same way training data was cleaned,
    then returns (predicted_intent, confidence_score, top_3_list).

    confidence_score is the model's probability/decision-based score for
    the predicted class -- higher means the model is more sure.
    """
    cleaned = clean_text(raw_text)

    # LinearSVC doesn't have predict_proba by default, so we handle both
    # probability-based models (LogisticRegression/NaiveBayes) and
    # decision-function-based models (LinearSVC) gracefully.
    predicted = model.predict([cleaned])[0]

    top3 = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_
        ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top3 = ranked[:3]
        confidence = dict(ranked)[predicted]
    elif hasattr(model.named_steps["clf"], "decision_function"):
        # Convert raw decision-function scores into a pseudo-confidence via
        # a softmax, purely for display purposes in the demo.
        import numpy as np
        scores = model.decision_function([cleaned])[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        classes = model.classes_
        ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top3 = ranked[:3]
        confidence = dict(ranked)[predicted]
    else:
        confidence = None

    return predicted, confidence, top3


def print_prediction(raw_text, predicted, confidence, top3):
    print(f"\nCommand entered : {raw_text}")
    print(f"Predicted action: {predicted}")
    print(f"{ACTION_DESCRIPTIONS.get(predicted, '')}")
    if confidence is not None:
        print(f"Confidence      : {confidence * 100:.1f}%")
    if top3:
        print("Top 3 candidates:")
        for intent, score in top3:
            print(f"   - {intent:16s} {score * 100:5.1f}%")


def main():
    model = load_model()

    # Single-command mode: `python src/predict.py "your command here"`
    if len(sys.argv) > 1:
        raw_text = " ".join(sys.argv[1:])
        predicted, confidence, top3 = predict_command(model, raw_text)
        print_prediction(raw_text, predicted, confidence, top3)
        return

    # Interactive mode
    print("=" * 60)
    print("Voice/Text Command Classifier - Interactive Demo")
    print("Type a Pidgin or English command (e.g. 'abeg check my balance')")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            raw_text = input("\nEnter command > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if raw_text.lower() in ("exit", "quit", ""):
            if raw_text.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            continue

        predicted, confidence, top3 = predict_command(model, raw_text)
        print_prediction(raw_text, predicted, confidence, top3)


if __name__ == "__main__":
    main()
