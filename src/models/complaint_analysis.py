
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "day3_complaint_analysis.txt"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "top_complaints.png"


def identify_theme(text):
    text = str(text).lower()

    if any(word in text for word in ["late", "delayed", "delay", "on time"]):
        if "on time" in text:
            return "Delivery timeliness (positive)"
        return "Late or delayed delivery"

    if any(word in text for word in ["cold", "soggy", "fresh"]):
        if "cold" in text or "soggy" in text:
            return "Food temperature or quality"
        return "Food freshness"

    if any(word in text for word in ["price", "worth", "expensive", "value"]):
        return "Price and value"

    if any(word in text for word in ["service", "experience"]):
        return "Service and overall experience"

    if any(word in text for word in ["taste", "delicious", "amazing", "great"]):
        return "Taste and food satisfaction"

    if any(word in text for word in ["okay", "acceptable", "fine", "special"]):
        return "Average or neutral experience"

    return "Other"


def main():
    reviews = pd.read_csv(INPUT_PATH)

    # Focus on rating-derived negative reviews.
    negative_reviews = reviews[reviews["sentiment"] == "negative"].copy()

    negative_reviews["complaint_theme"] = negative_reviews[
        "review_text"
    ].apply(identify_theme)

    theme_counts = negative_reviews["complaint_theme"].value_counts()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("DAY 3 — NEGATIVE REVIEW THEMES\n")
        file.write("=" * 45 + "\n")
        file.write(f"Total reviews: {len(reviews)}\n")
        file.write(f"Negative reviews: {len(negative_reviews)}\n\n")
        file.write("Complaint theme counts:\n")
        file.write(theme_counts.to_string())
        file.write("\n\nSample negative reviews by theme:\n")

        for theme in theme_counts.index:
            examples = negative_reviews.loc[
                negative_reviews["complaint_theme"] == theme,
                "review_text",
            ].drop_duplicates().head(3)

            file.write(f"\n{theme}:\n")
            for example in examples:
                file.write(f"- {example}\n")

    plot_counts = theme_counts.sort_values()

    plt.figure(figsize=(10, 6))
    plot_counts.plot(kind="barh")
    plt.title("Themes in Negative Customer Reviews")
    plt.xlabel("Number of reviews")
    plt.ylabel("Complaint theme")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print("=" * 60)
    print("DAY 3: NEGATIVE REVIEW THEME ANALYSIS")
    print("=" * 60)
    print(f"Total reviews: {len(reviews)}")
    print(f"Negative reviews: {len(negative_reviews)}")
    print("\nComplaint themes:")
    print(theme_counts.to_string())
    print("\nSaved:", REPORT_PATH)
    print("Saved:", FIGURE_PATH)


if __name__ == "__main__":
    main()
