# Uber Eats Operational & Customer Intelligence System

## Data Dictionary

All datasets in this project are synthetically generated for educational and analytical purposes. They do not contain real Uber Eats customer or operational data.

### 1. customers.csv

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `age` | Customer age in years |
| `city` | Customer's city |
| `signup_date` | Date the customer registered |
| `membership_type` | Basic, Plus, or Premium membership |
| `preferred_cuisine` | Customer's preferred cuisine |

### 2. restaurants.csv

| Column | Description |
|---|---|
| `restaurant_id` | Unique restaurant identifier |
| `restaurant_name` | Synthetic restaurant name |
| `city` | Restaurant's city |
| `cuisine` | Restaurant cuisine category |
| `average_rating` | Restaurant rating from 3 to 5 |
| `average_prep_time_min` | Average food preparation time in minutes |
| `is_active` | Whether the restaurant is active |

### 3. drivers.csv

| Column | Description |
|---|---|
| `driver_id` | Unique driver identifier |
| `vehicle_type` | Bike, Scooter, or Car |
| `experience_months` | Driver experience in months |
| `driver_rating` | Driver rating from 3.2 to 5 |

### 4. orders.csv

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `customer_id` | Customer who placed the order |
| `restaurant_id` | Restaurant receiving the order |
| `order_timestamp` | Date and time of the order |
| `order_hour` | Hour of day when the order was placed |
| `day_of_week` | Day of week when the order was placed |
| `order_value` | Food subtotal before delivery fee and discount |
| `delivery_fee` | Delivery fee charged |
| `discount_amount` | Discount applied |
| `distance_km` | Delivery distance in kilometres |
| `order_status` | Delivered, Cancelled, or Failed |

### 5. deliveries.csv

Contains delivery records for delivered orders only.

| Column | Description |
|---|---|
| `delivery_id` | Unique delivery identifier |
| `order_id` | Associated order |
| `driver_id` | Assigned driver |
| `traffic_level` | Low, Medium, or High traffic |
| `weather` | Clear, Rain, or Storm |
| `prep_time_min` | Food preparation time in minutes |
| `delivery_time_min` | Simulated delivery duration in minutes |
| `delivery_status` | Delivery status |

### 6. reviews.csv

Contains a sample of reviews for delivered orders.

| Column | Description |
|---|---|
| `review_id` | Unique review identifier |
| `order_id` | Associated order |
| `customer_id` | Customer who submitted the review |
| `restaurant_id` | Reviewed restaurant |
| `rating` | Review rating from 1 to 5 |
| `review_text` | Synthetic review text |
| `review_date` | Date associated with the review |

### 7. payments.csv

| Column | Description |
|---|---|
| `payment_id` | Unique payment record identifier |
| `order_id` | Associated order |
| `payment_method` | UPI, Card, Wallet, or Cash |
| `payment_amount` | Order value plus delivery fee minus discount |
| `payment_status` | Successful, Failed, or Refunded |

## Dataset Relationships

- `customers.customer_id` → `orders.customer_id`
- `restaurants.restaurant_id` → `orders.restaurant_id`
- `orders.order_id` → `deliveries.order_id`
- `drivers.driver_id` → `deliveries.driver_id`
- `orders.order_id` → `reviews.order_id`
- `orders.order_id` → `payments.order_id`

## Data Generation and Validation

The datasets are generated using Python, NumPy, and pandas with a fixed random seed for reproducibility. Initial validation checks cover missing values, duplicate primary IDs, and foreign-key relationships.

The datasets are intended for learning, exploratory data analysis, and machine learning experimentation. Synthetic patterns should not be interpreted as real-world business findings.
