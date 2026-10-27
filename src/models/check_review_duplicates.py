
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"


def main():
    reviews = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("REPEATED REVIEW AND LABEL CHECK")
    print("=" * 60)

    print(f"Total review rows: {len(reviews)}")
    print(f"Unique original texts: {reviews['review_text'].nunique()}")
    print(f"Unique cleaned texts: {reviews['cleaned_text'].nunique()}")

    # Count how many different sentiment labels each text receives.
    labels_per_text = reviews.groupby("cleaned_text")["sentiment"].nunique()
    inconsistent = labels_per_text[labels_per_text > 1]

    print(f"\nTexts associated with multiple sentiment labels: {len(inconsistent)}")

    print("\nReview-text frequency:")
    print(
        reviews["cleaned_text"]
        .value_counts()
        .rename_axis("cleaned_text")
        .reset_index(name="count")
        .to_string(index=False)
    )

    if len(inconsistent):
        print("\nExamples of inconsistent labels:")
        print(
            reviews[reviews["cleaned_text"].isin(inconsistent.index)]
            [["review_text", "rating", "sentiment"]]
            .drop_duplicates()
            .to_string(index=False)
        )
    else:
        print("\nEach unique cleaned text has a consistent sentiment label.")

    print("\nCheck complete.")


if __name__ == "__main__":
    main()
