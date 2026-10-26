# Day 2 — Data Preparation and Customer/Restaurant Segmentation

## 1. Objective

The objective of Day 2 was to prepare customer and restaurant features and apply K-Means and DBSCAN clustering to identify potential business segments.

## 2. Data Preparation and Feature Engineering

The following datasets were generated from the synthetic Uber Eats data:

- Customer feature dataset: 1,500 customers and 14 columns.
- Restaurant feature dataset: 100 restaurants and 14 columns.

Customer features include age, total orders, average order value, total spending, ordering days, weekend orders, late-night orders, average rating given, and review count.

Restaurant features include average rating, average preparation time, total orders, revenue, average order value, cancellation rate, average delivery time, review average rating, and review count.

Categorical features were one-hot encoded, numerical features were standardized, and identifier columns were excluded from model training.

## 3. K-Means Results

### Customer Segmentation

The silhouette score was highest at k = 2, with a score of 0.2100.

| Metric | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Customers | 629 | 871 |
| Average total orders | 5.01 | 2.12 |
| Average order value | 188.48 | 171.55 |
| Average total spending | 939.71 | 381.94 |
| Average review count | 3.14 | 1.12 |

**Interpretation:** Cluster 0 represents higher-frequency customers with greater total spending. Cluster 1 represents lower-frequency customers with lower total spending.

Potential business actions include loyalty incentives for frequent customers and controlled re-engagement campaigns for lower-frequency customers.

### Restaurant Segmentation

The silhouette score was highest at k = 2, with a score of 0.1547.

| Metric | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Restaurants | 55 | 45 |
| Average total orders | 45.78 | 55.16 |
| Average revenue | 8,303.77 | 10,378.60 |
| Average preparation time (minutes) | 28.38 | 19.31 |
| Average delivery time (minutes) | 54.43 | 46.26 |
| Cancellation rate | 7% | 5% |

**Interpretation:** Cluster 1 has higher average order volume and revenue, shorter preparation and delivery times, and a lower cancellation rate. Further investigation could identify operational practices associated with these differences.

Revenue is reported in the synthetic dataset's currency units; the currency should be specified before external reporting.

## 4. DBSCAN Results

| Dataset | Selected eps | Clusters excluding noise | Noise points |
|---|---:|---:|---:|
| Customers | 1.5 | 3 | 1,460 |
| Restaurants | 2.5 | 2 | 70 |

DBSCAN marked approximately 97.3% of customers and 70% of restaurants as noise with the selected settings.

**Limitation:** Although DBSCAN executed successfully, the high proportion of noise points limits its immediate usefulness for business segmentation. Further parameter tuning and evaluation are needed. Noise points should not automatically be interpreted as errors or unusual business behavior.

## 5. Evaluation and Limitations

- The highest tested K-Means silhouette score occurred at k = 2 for both datasets.
- Customer clustering achieved a silhouette score of 0.2100.
- Restaurant clustering achieved a silhouette score of 0.1547.
- These scores suggest limited cluster separation; the resulting segments should be treated as exploratory rather than definitive.
- The Elbow plots should also be reviewed when choosing the number of clusters.
- DBSCAN results require additional tuning because many records were classified as noise.
- Business interpretations are based on synthetic data and should not be treated as validated real-world findings.

## 6. Generated Deliverables

- Feature-engineered customer and restaurant datasets.
- K-Means and DBSCAN model artifacts for both datasets.
- Clustered datasets and K-Means cluster profiles.
- Elbow plots, silhouette plots, and PCA visualizations.

## 7. Conclusion

Day 2 successfully implemented feature engineering, preprocessing, K-Means clustering, DBSCAN clustering, cluster evaluation, and PCA visualization for customers and restaurants. Initial K-Means results reveal potential differences in customer purchasing frequency and restaurant operational performance. Further validation and DBSCAN tuning are recommended before using these segments to guide business decisions.
