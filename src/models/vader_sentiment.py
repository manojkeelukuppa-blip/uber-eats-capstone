
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "day3_vader_evaluation.txt"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "vader_confusion_matrix.png"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_with_vader.csv"


def classify_vader(compound):
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    reviews = pd.read_csv(INPUT_PATH)

    # VADER requires its lexicon. Download it once if it is not available.
    import nltk

    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        analyzer = SentimentIntensityAnalyzer()

    scores = reviews["review_text"].fillna("").apply(
        analyzer.polarity_scores
    )

    reviews["vader_compound"] = scores.apply(lambda item: item["compound"])
    reviews["vader_sentiment"] = reviews["vader_compound"].apply(
        classify_vader
    )

    y_true = reviews["sentiment"]
    y_pred = reviews["vader_sentiment"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=["negative", "neutral", "positive"],
        zero_division=0,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reviews.to_csv(OUTPUT_PATH, index=False)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("DAY 3 — VADER SENTIMENT EVALUATION\n")
        file.write("=" * 45 + "\n")
        file.write("Reference labels: rating-derived sentiment\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
        file.write(report)

    display = ConfusionMatrixDisplay(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["negative", "neutral", "positive"],
        ),
        display_labels=["Negative", "Neutral", "Positive"],
    )
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title("VADER Sentiment Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(FIGURE_PATH, dpi=150)
    plt.close(display.figure_)

    print("=" * 60)
    print("DAY 3: VADER SENTIMENT RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("\nVADER prediction distribution:")
    print(reviews["vader_sentiment"].value_counts())
    print("\nSaved:", OUTPUT_PATH)
    print("Saved:", REPORT_PATH)
    print("Saved:", FIGURE_PATH)


if __name__ == "__main__":
    main()
