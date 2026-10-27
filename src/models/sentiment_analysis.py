
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "day3_sentiment_evaluation.txt"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "sentiment_confusion_matrix.png"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    reviews = pd.read_csv(INPUT_PATH)
    reviews = reviews.dropna(subset=["cleaned_text", "sentiment"])

    X = reviews["cleaned_text"].astype(str)
    y = reviews["sentiment"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=1000,
                sublinear_tf=True,
            ),
        ),
        ("classifier", MultinomialNB()),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    report = classification_report(
        y_test,
        predictions,
        labels=["negative", "neutral", "positive"],
        zero_division=0,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("DAY 3 — SENTIMENT CLASSIFICATION EVALUATION\n")
        file.write("=" * 55 + "\n")
        file.write(f"Total reviews: {len(reviews)}\n")
        file.write(f"Training reviews: {len(X_train)}\n")
        file.write(f"Testing reviews: {len(X_test)}\n")
        file.write("Model: TF-IDF + Multinomial Naive Bayes\n")
        file.write("Target: Rating-derived sentiment labels\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
        file.write("Classification report:\n")
        file.write(report)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["negative", "neutral", "positive"],
        display_labels=["Negative", "Neutral", "Positive"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("Sentiment Classification Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(FIGURE_PATH, dpi=150)
    plt.close(display.figure_)

    print("=" * 60)
    print("DAY 3: SENTIMENT CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Training reviews: {len(X_train)}")
    print(f"Testing reviews: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("\nSaved model:", MODEL_PATH)
    print("Saved report:", REPORT_PATH)
    print("Saved confusion matrix:", FIGURE_PATH)


if __name__ == "__main__":
    main()
