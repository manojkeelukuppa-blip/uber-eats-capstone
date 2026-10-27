
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MATRIX_PATH = OUTPUT_DIR / "review_tfidf_matrix.csv"
FEATURE_NAMES_PATH = OUTPUT_DIR / "tfidf_feature_names.csv"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    reviews = pd.read_csv(INPUT_PATH)

    if "cleaned_text" not in reviews.columns:
        raise ValueError("The dataset must contain a cleaned_text column.")

    # Fit on the available reviews for this exploratory vocabulary analysis.
    # For later predictive evaluation, fit TF-IDF on training data only.
    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=1000,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(
        reviews["cleaned_text"].fillna("")
    )

    feature_names = vectorizer.get_feature_names_out()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the sparse matrix as a CSV with review identifiers.
    matrix_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
    )
    matrix_df.insert(0, "review_id", reviews["review_id"].values)
    matrix_df.to_csv(MATRIX_PATH, index=False)

    pd.DataFrame({"term": feature_names}).to_csv(
        FEATURE_NAMES_PATH,
        index=False,
    )

    # Save the vectorizer so the same vocabulary can be reused later.
    import joblib
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # Rank terms by their average TF-IDF score across reviews.
    mean_scores = tfidf_matrix.mean(axis=0).A1
    ranked_terms = (
        pd.DataFrame({"term": feature_names, "mean_tfidf": mean_scores})
        .sort_values("mean_tfidf", ascending=False)
    )

    print("=" * 60)
    print("DAY 3: TF-IDF FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Number of reviews: {tfidf_matrix.shape[0]:,}")
    print(f"Number of TF-IDF features: {tfidf_matrix.shape[1]:,}")
    print(f"Matrix density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4%}")

    print("\nTop 20 terms/phrases by mean TF-IDF:")
    print(ranked_terms.head(20).to_string(index=False))

    print("\nSaved files:")
    print(MATRIX_PATH)
    print(FEATURE_NAMES_PATH)
    print(VECTORIZER_PATH)

    print("\nTF-IDF extraction complete.")


if __name__ == "__main__":
    main()
