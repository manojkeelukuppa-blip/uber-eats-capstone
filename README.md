# Uber Eats Operational & Customer Intelligence System

## Project Overview

This project develops a data-driven operational and customer intelligence system for a food delivery platform inspired by Uber Eats. It uses synthetic data to explore customer behaviour, restaurant performance, delivery operations, customer reviews, and order demand.

The project is planned as a seven-day machine learning capstone covering data generation, exploratory analysis, customer segmentation, sentiment analysis, delivery-time prediction, demand forecasting, and business recommendations.

**Current status:** Day 1 — Project setup, synthetic data generation, data validation, and documentation.

> **Disclaimer:** All datasets are synthetically generated for educational purposes. They do not represent actual Uber Eats data or verified business performance.

## Project Objectives

- Generate realistic, reproducible synthetic operational datasets.
- Establish a structured and version-controlled Python project.
- Document dataset fields and relationships.
- Prepare for customer segmentation using clustering.
- Prepare for customer review sentiment analysis.
- Prepare for delivery-time prediction and demand forecasting.
- Translate future analytical findings into actionable business recommendations.

## Technology Stack

- Python
- NumPy and pandas
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook
- NLTK and VADER Sentiment
- Statsmodels
- Git and GitHub

## Project Structure

```text
uber-eats-ml-capstone/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   │   └── generate_data.py
│   ├── features/
│   └── models/
├── models/
├── reports/
│   └── figures/
├── tests/
├── docs/
│   └── data_dictionary.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Synthetic Datasets

The generator creates seven CSV files in `data/raw/`.

| Dataset | Rows generated | Description |
|---|---:|---|
| customers.csv | 1,500 | Customer demographics, location, membership, and cuisine preferences |
| restaurants.csv | 100 | Restaurant location, cuisine, ratings, preparation time, and activity |
| drivers.csv | 200 | Driver vehicle type, experience, and rating |
| orders.csv | 5,000 | Order timestamps, values, fees, discounts, distance, and status |
| deliveries.csv | 4,538 | Delivery records, traffic, weather, preparation time, and delivery duration |
| reviews.csv | 2,949 | Customer ratings and synthetic review text |
| payments.csv | 5,000 | Payment methods, amounts, and payment statuses |

The row counts above are from the current successful generation run.

## Dataset Relationships

- `customers.customer_id` → `orders.customer_id`
- `restaurants.restaurant_id` → `orders.restaurant_id`
- `orders.order_id` → `deliveries.order_id`
- `drivers.driver_id` → `deliveries.driver_id`
- `orders.order_id` → `reviews.order_id`
- `orders.order_id` → `payments.order_id`

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for column-level definitions.

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd uber-eats-ml-capstone
```

Replace `<your-repository-url>` with the actual repository URL if the project is hosted remotely.

### 2. Create and activate a virtual environment

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Generate the synthetic datasets

From the project root, run:

```bash
python src/data/generate_data.py
```

The generated CSV files will be saved in `data/raw/`.

## Data Validation

Initial validation was performed on all seven datasets.

- No missing values were detected in the generated datasets.
- No duplicate primary IDs were detected.
- Customer, restaurant, driver, order, delivery, review, and payment relationships passed the implemented foreign-key checks.
- Delivery records were verified to reference delivered orders.

These checks validate basic data consistency; they do not establish that the synthetic data accurately represents real-world behaviour.

## Planned Project Roadmap

| Day | Planned work |
|---|---|
| Day 1 | Git setup, synthetic data generation, validation, data dictionary, prompt engineering, and documentation |
| Day 2 | Feature engineering, scaling, clustering, and customer segment interpretation |
| Day 3 | Review preprocessing, sentiment classification, model evaluation, and sentiment insights |
| Day 4 | Delivery-time or tip prediction using regression models |
| Day 5 | Hourly demand analysis and time-series forecasting |
| Day 6 | Pipeline integration, business insights, recommendations, and limitations |
| Day 7 | Final presentation, documentation, and Git workflow |

## Reproducibility

The synthetic data generator uses a fixed random seed to make data generation reproducible under the same software environment. The generated datasets can be recreated by running the generator.

## Limitations

- The datasets are synthetic and may not capture the complexity of real delivery operations.
- Simulated relationships may influence model performance and analytical conclusions.
- Results should be treated as educational demonstrations, not production-ready business evidence.
- Any future machine learning model will require appropriate evaluation before practical use.

## Author and Project Information

**Project:** Uber Eats Operational & Customer Intelligence System  
**Type:** Machine Learning Capstone  
**Status:** Day 1 completed in progress; subsequent project stages are planned.
