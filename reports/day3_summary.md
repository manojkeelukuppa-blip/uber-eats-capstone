# Day 3 — NLP Sentiment Analysis

## 1. Objective

Analyze customer reviews using natural language processing (NLP), classify review sentiment, compare a machine-learning classifier with VADER, and identify common negative-review themes.

## 2. Dataset

- Source: Synthetic Uber Eats review dataset
- Total reviews analyzed: 2,949
- Positive reviews: 2,041 (69.21%)
- Neutral reviews: 502 (17.02%)
- Negative reviews: 406 (13.77%)
- Missing review text or rating after filtering: 0

Sentiment reference labels were derived from ratings: ratings 1–2 were negative, rating 3 was neutral, and ratings 4–5 were positive.

## 3. NLP preprocessing

The preprocessing pipeline performs:
- Lowercasing and removal of URLs, punctuation, and digits
- Tokenization
- Stop-word removal while preserving important negation words
- Lemmatization using NLTK WordNet
- Creation of cleaned review text

The processed dataset is saved as `data/processed/cleaned_reviews.csv`.

## 4. TF-IDF feature extraction

TF-IDF was used to represent review text numerically, with unigrams and bigrams.

- Reviews processed: 2,949
- Features extracted: 60
- Matrix density: 10.6968%

Common terms by mean TF-IDF included food, delivery, taste, and experience. These are descriptive vocabulary weights and are not necessarily the most predictive features.

## 5. Sentiment classification

### Multinomial Naive Bayes

- Training reviews: 2,359
- Testing reviews: 590
- Accuracy: 100.00%
- Macro F1-score: 100.00%

### VADER

- Accuracy: 83.42%
- Macro F1-score: 66.65%
- Negative-class F1-score: 0.71
- Neutral-class F1-score: 0.35
- Positive-class F1-score: 0.94

VADER identified positive reviews well but had difficulty identifying neutral reviews.

## 6. Negative-review themes

The rule-based complaint analysis identified 406 negative reviews:

| Theme | Reviews | Share of negative reviews |
|---|---:|---:|
| Late or delayed delivery | 219 | 53.94% |
| Price and value | 95 | 23.40% |
| Service and overall experience | 92 | 22.66% |

Delivery timeliness was the most frequent complaint theme in this synthetic dataset.

## 7. Business recommendations

1. Investigate causes of late or delayed deliveries.
2. Monitor delivery-time reliability and communicate delays to customers.
3. Review customer feedback related to price and perceived value.
4. Track service-related feedback and use it to identify improvement opportunities.

These recommendations are exploratory and should be validated with real operational data before business decisions are made.

## 8. Limitations

- The review dataset contains only 12 unique review-text patterns. Repeated texts occur in both training and test sets, so the perfect Naive Bayes score likely reflects memorization and should not be interpreted as real-world generalization.
- Reference sentiment labels are derived from ratings, not independently annotated sentiment.
- VADER is a lexicon-based method and may not interpret every review in context.
- Complaint themes use simple keyword rules rather than a trained topic model. The categories are not exhaustive.
- The dataset is synthetic and may not reflect the diversity of actual customer feedback.

## 9. Generated artifacts

- `data/processed/cleaned_reviews.csv`
- `data/processed/review_tfidf_matrix.csv`
- `data/processed/tfidf_feature_names.csv`
- `data/processed/reviews_with_vader.csv`
- `models/tfidf_vectorizer.joblib`
- `models/sentiment_model.joblib`
- `reports/day3_sentiment_evaluation.txt`
- `reports/day3_vader_evaluation.txt`
- `reports/day3_complaint_analysis.txt`
- `reports/figures/sentiment_confusion_matrix.png`
- `reports/figures/vader_confusion_matrix.png`
- `reports/figures/top_complaints.png`

## 10. Conclusion

Day 3 implemented text preprocessing, TF-IDF feature extraction, Naive Bayes sentiment classification, VADER comparison, and rule-based complaint analysis. The results highlight delivery timeliness as the leading negative-review theme in the synthetic data. Model evaluation limitations are documented to avoid overstating predictive performance.