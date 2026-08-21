"""
preprocess.py
--------------
Text cleaning utilities shared by both training and prediction.

IMPORTANT: The SAME cleaning function must be used at training time and at
prediction time. If they differ, the model will see different-looking text
at inference than what it learned from, and accuracy will silently drop.
That's why this logic lives in its own small module that both train.py and
predict.py import.
"""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Normalizes a raw command string before it is fed to the vectorizer.

    Steps:
    1. Lowercase everything (so "Send" and "send" are treated the same).
    2. Remove punctuation/special characters (keep letters, numbers, spaces).
    3. Collapse multiple spaces into one and strip leading/trailing spaces.

    We deliberately keep it simple (no stemming/lemmatization) because:
    - Pidgin English doesn't have reliable NLP tools (no stemmer trained on it)
    - Short commands (3-8 words) don't benefit much from stemming
    - TF-IDF + n-grams already captures a lot of the useful signal
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Remove anything that isn't a letter, digit, apostrophe, or whitespace
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the commands dataset CSV and applies text cleaning.

    Returns a DataFrame with columns:
        - command        : original raw text
        - command_clean   : cleaned text (used for training)
        - intent          : the label
    """
    df = pd.read_csv(csv_path)

    # Basic sanity checks so failures are easy to debug
    assert "command" in df.columns, "CSV must have a 'command' column"
    assert "intent" in df.columns, "CSV must have an 'intent' column"

    # Drop any completely empty rows just in case
    df = df.dropna(subset=["command", "intent"]).reset_index(drop=True)

    df["command_clean"] = df["command"].apply(clean_text)
    return df
