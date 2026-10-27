
from pathlib import Path
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"

# Keep negation words because they can reverse sentiment.
NEGATION_WORDS = {"no", "not", "nor", "never", "n't"}

STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    """Clean, tokenize, remove stop words, and lemmatize review text."""
    text = str(text).lower()

    # Remove URLs, punctuation, and digits.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization.
    tokens = text.split()

    # Stop-word removal and lemmatization.
    processed_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS
    ]

    return " ".join(processed_tokens)


def main():
    reviews = pd.read_csv(INPUT_PATH)

    # Remove rows without the fields needed for analysis.
    reviews = reviews.dropna(subset=["rating", "review_text"]).copy()

    # Explicit NLP preprocessing.
    reviews["cleaned_text"] = reviews["review_text"].apply(preprocess_text)

    # Create rating-derived reference labels.
    reviews["sentiment"] = reviews["rating"].apply(
        lambda rating: (
            "negative" if rating <= 2
            else "neutral" if rating == 3
            else "positive"
        )
    )

    # Remove empty results if any exist.
    reviews = reviews[reviews["cleaned_text"].str.strip().ne("")].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("DAY 3: NLP TEXT PREPROCESSING")
    print("=" * 60)
    print(f"Rows saved: {len(reviews)}")
    print(f"Missing cleaned text: {reviews['cleaned_text'].isna().sum()}")
    print("\nSentiment distribution:")
    print(reviews["sentiment"].value_counts())
    print("\nOriginal and processed examples:")
    print(reviews[["review_text", "cleaned_text", "sentiment"]].head(10).to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
