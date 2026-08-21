"""
evaluate.py
------------
Loads the already-trained model and produces visual/inspectable evaluation
artifacts:
    - results/confusion_matrix_test_split.png   (template-style test set)
    - results/confusion_matrix_hard_test.png    (hand-written unseen set)
    - results/metrics.json (already written by train.py, re-used here)

Run AFTER train.py:
    python src/evaluate.py
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNG files
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "command_classifier.joblib")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "data", "test_split.csv")
HARD_TEST_PATH = os.path.join(BASE_DIR, "data", "hard_test_set.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    """
    Builds and saves a confusion matrix heatmap image.
    Rows = true intent, Columns = predicted intent.
    A perfect classifier would have all its values on the diagonal.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 9))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to: {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    labels = sorted(model.classes_)

    # ---- Evaluate on the template-style held-out test split ----
    if os.path.exists(TEST_SPLIT_PATH):
        test_df = pd.read_csv(TEST_SPLIT_PATH)
        y_pred = model.predict(test_df["command_clean"])
        plot_confusion_matrix(
            test_df["intent"], y_pred, labels,
            title="Confusion Matrix - Template Test Split",
            out_path=os.path.join(RESULTS_DIR, "confusion_matrix_test_split.png"),
        )

    # ---- Evaluate on the hand-written hard/unseen test set ----
    if os.path.exists(HARD_TEST_PATH):
        hard_df = pd.read_csv(HARD_TEST_PATH)
        hard_df["command_clean"] = hard_df["command"].apply(clean_text)
        y_pred_hard = model.predict(hard_df["command_clean"])
        plot_confusion_matrix(
            hard_df["intent"], y_pred_hard, labels,
            title="Confusion Matrix - Hard/Unseen Test Set",
            out_path=os.path.join(RESULTS_DIR, "confusion_matrix_hard_test.png"),
        )

    print("\nEvaluation complete. Check the results/ folder for PNG images and reports.")


if __name__ == "__main__":
    main()
