# Uber Eats Operational & Customer Intelligence System

## Project Overview

The **Uber Eats Operational & Customer Intelligence System** is a machine learning capstone project focused on analysing customer behaviour, restaurant performance, delivery operations, customer feedback, and hourly order demand in a food delivery marketplace.

The project uses synthetically generated datasets to demonstrate an end-to-end data analytics and machine learning workflow, including:

- Data generation and validation
- Exploratory data analysis and feature engineering
- Customer and restaurant segmentation
- Customer review sentiment analysis
- Delivery-time prediction using regression
- Hourly demand forecasting
- Integrated business insights and recommendations

**Project status:** Days 1–6 completed. Day 7 final presentation and Git finalization are pending.

> **Disclaimer:** All datasets used in this project are synthetically generated for educational purposes. They are not actual Uber Eats operational data and do not represent verified business performance.

## Project Objectives

1. Analyse customer purchasing behaviour and identify meaningful customer segments.
2. Evaluate restaurant performance using operational and order-related features.
3. Analyse customer reviews and identify sentiment and recurring complaints.
4. Predict delivery duration using machine learning regression models.
5. Analyse hourly order demand and forecast future demand.
6. Integrate analytical findings into actionable business recommendations.

## Technology Stack

| Category | Tools and Libraries |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Clustering | K-Means, DBSCAN, PCA |
| NLP and Sentiment Analysis | NLTK, VADER, TF-IDF |
| Regression | Linear Regression, Ridge, Lasso, Random Forest |
| Time Series | Statsmodels, ARIMA, Prophet |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Joblib |
| Development | VS Code, Jupyter |
| Version Control | Git, GitHub |

## Dataset Description

Seven synthetic datasets were generated for the project.

| Dataset | Records | Description |
|---|---:|---|
| customers.csv | 1,500 | Customer demographics, location, membership, and cuisine preferences |
| restaurants.csv | 100 | Restaurant location, cuisine, ratings, preparation time, and activity |
| drivers.csv | 200 | Driver vehicle type, experience, and rating |
| orders.csv | 5,000 | Order timestamps, values, fees, discounts, distance, and status |
| deliveries.csv | 4,538 | Delivery records, traffic, weather, preparation time, and delivery duration |
| reviews.csv | 2,949 | Customer ratings and synthetic review text |
| payments.csv | 5,000 | Payment methods, amounts, and payment statuses |

### Dataset Relationships

- `customers.customer_id` → `orders.customer_id`
- `restaurants.restaurant_id` → `orders.restaurant_id`
- `orders.order_id` → `deliveries.order_id`
- `drivers.driver_id` → `deliveries.driver_id`
- `orders.order_id` → `reviews.order_id`
- `orders.order_id` → `payments.order_id`

Refer to [`docs/data_dictionary.md`](docs/data_dictionary.md) for field definitions and dataset details.

## Project Structure

```text
uber-eats-ml-capstone/
│
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── restaurants.csv
│   │   ├── drivers.csv
│   │   ├── orders.csv
│   │   ├── deliveries.csv
│   │   ├── reviews.csv
│   │   └── payments.csv
│   │
│   └── processed/
│       ├── customer_features.csv
│       ├── restaurant_features.csv
│       ├── customer_clustered.csv
│       ├── restaurant_clustered.csv
│       ├── cleaned_reviews.csv
│       ├── delivery_time_features.csv
│       ├── hourly_demand.csv
│       └── ...
│
├── docs/
│   ├── data_dictionary.md
│   └── prompt_engineering.md
│
├── models/
│   ├── customer_kmeans.joblib
│   ├── customer_dbscan.joblib
│   ├── restaurant_kmeans.joblib
│   ├── restaurant_dbscan.joblib
│   ├── sentiment_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── delivery_time_best_model.joblib
│
├── notebooks/
│
├── reports/
│   ├── day2_segmentation_report.md
│   ├── day3_summary.md
│   ├── day4_summary.md
│   ├── day5_demand_forecasting_report.md
│   ├── day6_integrated_business_insights.md
│   └── figures/
│
├── src/
│   ├── data/
│   │   └── generate_data.py
│   ├── features/
│   │   ├── feature_engineering.py
│   │   ├── text_preprocessing.py
│   │   ├── tfidf_features.py
│   │   ├── delivery_time_eda.py
│   │   ├── delivery_time_features.py
│   │   ├── delivery_time_prediction.py
│   │   ├── hourly_demand_analysis.py
│   │   ├── demand_decomposition.py
│   │   ├── demand_forecasting.py
│   │   └── integrate_pipeline.py
│   └── models/
│       ├── segmentation.py
│       ├── sentiment_analysis.py
│       └── vader_sentiment.py
│
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Implementation and Results

### Day 1 — Project Setup and Synthetic Data Generation

- Initialized the Git repository and created the feature branch.
- Established the project directory structure.
- Generated seven synthetic datasets.
- Performed initial data-quality and relationship validation.
- Created the data dictionary.
- Documented prompt engineering techniques, including P.T.C.F., few-shot, role-based, and structured-output prompting.

**Validation results:**
- All seven datasets were generated successfully.
- No missing values were detected in the initial generated datasets.
- No duplicate primary IDs were detected.
- Implemented foreign-key relationship checks passed.

### Day 2 — Customer and Restaurant Segmentation

Implemented feature engineering, scaling, K-Means clustering, DBSCAN clustering, PCA, and cluster evaluation.

**Customer segmentation:**

| Metric | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Customers | 629 | 871 |
| Average orders | 5.01 | 2.12 |
| Average order value | 188.48 | 171.55 |
| Average total spending | 939.71 | 381.94 |
| Average reviews | 3.14 | 1.12 |

- Selected K-Means with 2 clusters.
- Customer silhouette score: **0.2100**.

**Restaurant segmentation:**

| Metric | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Restaurants | 55 | 45 |
| Average orders | 45.78 | 55.16 |
| Average revenue | 8,303.77 | 10,378.60 |
| Average preparation time | 28.38 min | 19.31 min |
| Average delivery time | 54.43 min | 46.26 min |
| Cancellation rate | 7% | 5% |

- Selected K-Means with 2 clusters.
- Restaurant silhouette score: **0.1547**.

**Observation:** The customer clusters indicate differences in order frequency and spending. Restaurant clusters show differences in order volume, revenue, and operational efficiency.

DBSCAN was also evaluated. It classified a high proportion of observations as noise, so its results were treated as exploratory rather than as the primary segmentation solution.

### Day 3 — NLP and Sentiment Analysis

Performed text preprocessing, TF-IDF feature extraction, sentiment classification, VADER analysis, and complaint analysis.

**Sentiment distribution:**

| Sentiment | Reviews | Percentage |
|---|---:|---:|
| Positive | 2,041 | 69.21% |
| Neutral | 502 | 17.02% |
| Negative | 406 | 13.77% |
| **Total** | **2,949** | **100%** |

Sentiment labels were derived from customer ratings:
- Ratings 1–2: Negative
- Rating 3: Neutral
- Ratings 4–5: Positive

**Model evaluation:**

- TF-IDF with Multinomial Naive Bayes achieved 100% accuracy and macro F1 on the evaluated split.
- However, the synthetic review dataset contained only 12 unique text patterns. Repeated text patterns across training and testing may have caused memorization and inflated performance.
- VADER achieved 83.42% accuracy and 66.65% macro F1.

**Negative review themes:**

| Complaint Category | Reviews | Percentage of Negative Reviews |
|---|---:|---:|
| Late or delayed delivery | 219 | 53.94% |
| Price and value | 95 | 23.40% |
| Service and overall experience | 92 | 22.66% |

**Observation:** Delivery delays were the most frequent complaint category in the synthetic review data.

### Day 4 — Delivery-Time Prediction

Selected delivery-time prediction as the regression use case.

- Analysed 4,538 delivery records.
- Average delivery time: 50.68 minutes.
- Median delivery time: 49 minutes.
- Minimum delivery time: 11 minutes.
- Maximum delivery time: 137 minutes.

Distance, delivery fee, and preparation time showed notable relationships with delivery duration.

**Model comparison:**

| Model | MAE (minutes) | RMSE | R² |
|---|---:|---:|---:|
| Lasso Regression | 3.9076 | 5.0175 | 0.91232 |
| Linear Regression | 3.9112 | 5.0186 | 0.91228 |
| Ridge Regression | 3.9123 | 5.0204 | 0.91222 |
| Random Forest | 4.4608 | 5.7584 | 0.88452 |

**Selected model:** Lasso Regression.

The selected model achieved:
- Test MAE: **3.9076 minutes**
- Test RMSE: **5.0175 minutes**
- Test R²: **0.91232**

Random Forest feature importance highlighted distance, average preparation time, preparation time, delivery fee, traffic, and weather as useful predictive features.

**Observation:** Delivery distance and restaurant preparation time are important operational factors associated with delivery duration in the synthetic dataset.

### Day 5 — Hourly Demand Forecasting

Performed hourly demand analysis, decomposition, stationarity testing, ARIMA forecasting, and Prophet forecasting.

**Dataset overview:**
- Historical period: January 1 to June 30, 2025.
- Hourly observations: 4,343.
- Average demand: 1.151 orders per hour.
- Maximum observed demand: 6 orders per hour.

The Augmented Dickey-Fuller test returned:
- ADF statistic: **-65.5493**
- p-value: **less than 0.000001**

This provided evidence of stationarity under the ADF test.

**Forecast model comparison:**

| Model | MAE | RMSE | WAPE | Non-zero MAPE |
|---|---:|---:|---:|---:|
| ARIMA (1,0,1) | 0.8018 | 1.0224 | 70.81% | 30.60% |
| Prophet | 0.8056 | 1.0222 | 71.15% | 31.19% |
| Seasonal Naive (24h) | 0.9965 | 1.3217 | 88.01% | 55.09% |

**Observation:**
- ARIMA achieved the lowest MAE and WAPE.
- Prophet achieved a marginally lower RMSE.
- Both forecasting models outperformed the seasonal naive baseline on the reported metrics.

The generated 24-hour forecast is for July 1, 2025. It is a historical forecast based on the synthetic dataset, not a live demand prediction.

### Day 6 — Pipeline Integration and Business Insights

Integrated the outputs from segmentation, sentiment analysis, delivery-time prediction, and demand forecasting.

Created:

- `src/features/integrate_pipeline.py`
- `reports/day6_customer_segment_summary.csv`
- `reports/day6_restaurant_segment_summary.csv`
- `reports/day6_sentiment_summary.csv`
- `reports/day6_delivery_summary.json`
- `reports/day6_demand_summary.json`
- `reports/day6_integrated_business_insights.md`

The integration script successfully loaded the analytical outputs and generated a consolidated business insights report.

**Key business recommendations:**

1. **Customer engagement:** Develop targeted retention and loyalty strategies for customer segments with different ordering and spending patterns.
2. **Restaurant operations:** Identify restaurants with higher preparation and delivery times and investigate opportunities to improve operational efficiency.
3. **Delivery experience:** Prioritize reducing delivery delays through better preparation-time coordination and delivery planning.
4. **Customer feedback:** Monitor recurring complaints, especially delivery delays, and use review insights to guide service improvements.
5. **Demand planning:** Use hourly demand patterns to support driver allocation and operational planning, while validating forecasts before practical deployment.

Refer to [`reports/day6_integrated_business_insights.md`](reports/day6_integrated_business_insights.md) for the consolidated findings, recommendations, limitations, and future improvements.

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/manojkeelukuppa-blip/uber-eats-capstone.git
cd uber-eats-capstone
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Generate Synthetic Data

From the project root:

```bash
python src/data/generate_data.py
```

This generates the raw CSV datasets in `data/raw/`.

**Note:** Running the generator again may replace the existing raw datasets. If you want to preserve the current project outputs, back them up first.

### 5. Run the Analysis Scripts

The main implementation scripts are organized under `src/features/` and `src/models/`.

Examples:

```bash
python src/features/feature_engineering.py
python src/models/segmentation.py
python src/features/text_preprocessing.py
python src/features/tfidf_features.py
python src/models/sentiment_analysis.py
python src/models/vader_sentiment.py
python src/features/delivery_time_eda.py
python src/features/delivery_time_features.py
python src/features/delivery_time_prediction.py
python src/features/hourly_demand_analysis.py
python src/features/demand_decomposition.py
python src/features/demand_forecasting.py
```

Run the scripts from the project root and follow the required execution order and input dependencies. Existing processed datasets and saved models are included in the project.

### 6. Run the Integrated Pipeline

```bash
python src/features/integrate_pipeline.py
```

The integrated pipeline combines the outputs from the completed analytical stages and generates consolidated business summaries and insights.

## Reports and Visualizations

The `reports/` directory contains model evaluation results, analytical summaries, and visualizations.

Examples include:

- Customer and restaurant segmentation reports.
- Sentiment evaluation and complaint analysis.
- Delivery-time model comparison and feature importance.
- Demand forecasting evaluation and summaries.
- Actual-versus-predicted delivery-time plots.
- Customer and restaurant PCA visualizations.
- Sentiment confusion matrices.
- Hourly demand and forecast visualizations.
- Integrated business insights.

Visualizations are stored in `reports/figures/`.

## GitHub Repository

**Repository:** [Uber Eats Operational & Customer Intelligence System](https://github.com/manojkeelukuppa-blip/uber-eats-capstone)

The project uses Git for version control and a feature branch named `feature/uber-eats-pipeline` for development.

## Limitations

1. **Synthetic data:** The datasets are simulated and may not reflect real-world food delivery operations.
2. **Clustering quality:** The relatively low silhouette scores indicate that the customer and restaurant clusters have limited separation.
3. **DBSCAN performance:** A high percentage of records were classified as noise, limiting its usefulness for the current dataset.
4. **Sentiment generalization:** Repeated review patterns may have inflated TF-IDF model performance. The results should not be interpreted as evidence of robust real-world generalization.
5. **Regression generalization:** Delivery-time prediction results are based on synthetic relationships and require validation using real operational data.
6. **Forecasting accuracy:** The hourly demand series is sparse, and the WAPE values are relatively high. Forecasts should therefore be interpreted cautiously.
7. **Historical forecast:** The generated July 1, 2025 forecast is not a live operational forecast.
8. **Validation scope:** The project demonstrates an educational analytical workflow and is not a production-ready deployment.

## Future Improvements

- Validate the models using real-world operational data.
- Improve cluster quality through additional features and parameter tuning.
- Expand and diversify the review dataset to improve sentiment generalization.
- Evaluate sentiment models using more robust, independent test data.
- Explore additional delivery-time predictors and more extensive model validation.
- Apply rolling-origin validation and investigate seasonal forecasting approaches.
- Build an interactive dashboard for monitoring customer, restaurant, delivery, and demand metrics.
- Develop a production-ready pipeline with automated testing and monitoring.

## Author and Project Information

**Project:** Uber Eats Operational & Customer Intelligence System
**Project Type:** Machine Learning Capstone
**Domain:** Food Delivery and Marketplace Analytics
**Role:** Trainee Data Analyst – ML Focus
**Development Environment:** VS Code, Python, Git, and GitHub
**Project Progress:** Days 1–6 completed; Day 7 final presentation and Git finalization pending.
