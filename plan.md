# Churn Prediction Microservice Plan (Python + Aerospike Feature Store)

## Objective

Build a Python microservice for **churn prediction** using Aerospike Feature Store to manage features and XGBoost as the prediction model. The service will expose APIs to ingest data, retrieve features, score users, and trigger retention nudges based on risk scores and churn reasons.

---

## Features

### User Profile Features
* user_id
* account_age_days
* membership_duration
* loyalty_tier
* geo_location
* device_type
* preferred_payment_method
* language_preference

### User Behavior Features
* days_since_last_login
* days_since_last_purchase
* sessions_last_7days
* sessions_last_30days
* avg_session_duration_last_30days
* click_through_rate_last_10_sessions
* cart_abandonment_rate
* wishlist_adds_vs_purchases
* content_engagement_rate

### Transactional Features
* avg_order_value
* total_orders_last_6months
* purchase_frequency_last_90days
* time_since_last_high_value_purchase
* refund_rate
* subscription_payment_status
* discount_dependency_score
* category_spend_distribution

### Engagement Features
* push_notification_open_rate
* email_click_rate
* in_app_offer_click_rate
* response_time_to_promotions
* recent_retention_offer_response

### Support Features
* support_tickets_last_90days
* avg_ticket_resolution_time
* csat_score_last_interaction
* refund_requests

### Real Time Session Features
* current_session_clicks
* time_spent_on_checkout_page
* added_to_cart_but_not_bought_flag
* session_bounce_flag

### Derived or Aggregated Features
* churn_risk_score_trend
* engagement_velocity
* price_sensitivity_score
* customer_lifetime_value_bucket
* rolling_average_purchase_frequency
* long_term_vs_short_term_activity_ratio

---

## Triggers & Nudge Flow

### Triggers
- **Session Start**  
- **Checkout Abandonment**  
- **Scheduled Events** (e.g., everyday at 9 AM)  
- **Created Support Ticket**  

### Process
1. Each trigger **calls the Churn Prediction API**.  
2. API returns:  
   - **Churn score** (probability of churn)  
   - **Risk segment** (`low`, `medium`, `high`, `critical`)  
   - **Churn reasons** (one or more from enum):  
     ```
     INACTIVITY, CART_ABANDONMENT, LOW_ENGAGEMENT, PRICE_SENSITIVITY, 
     DELIVERY_ISSUES, PRODUCT_AVAILABILITY, PAYMENT_FAILURE
     ```
3. **Decision:**  
   - If risk is **high or critical**, invoke the **Nudge Service**.  

### Nudge Service
- Determines **nudge type(s)** based on churn score and churn reasons.  
- Possible nudges: **Push Notification, Email, Discount Coupon**  

---

## Initial Nudge Rule Mappings (Version 1)

**Rule 1**  
- **Churn Score:** 0.6 – 0.8  
- **Churn Reason:** INACTIVITY, DELIVERY_ISSUES  
- **Nudge Type:** Email  
- **Content Template:** Template 1  

**Rule 2**  
- **Churn Score:** 0.8 – 1.0  
- **Churn Reason:** CART_ABANDONMENT  
- **Nudge Type:** Push Notification + Discount Coupon  
- **Content Template:** Template 2  

**Rule 3**  
- **Churn Score:** 0.7 – 0.9  
- **Churn Reason:** LOW_ENGAGEMENT  
- **Nudge Type:** Email  
- **Content Template:** Template 3  

**Rule 4**  
- **Churn Score:** 0.6 – 0.75  
- **Churn Reason:** PRICE_SENSITIVITY  
- **Nudge Type:** Discount Coupon  
- **Content Template:** Template 4  

**Rule 5**  
- **Churn Score:** 0.85 – 1.0  
- **Churn Reason:** PAYMENT_FAILURE  
- **Nudge Type:** Push Notification + Email  
- **Content Template:** Template 5  

**Rule 6**  
- **Churn Score:** 0.65 – 0.8  
- **Churn Reason:** PRODUCT_AVAILABILITY  
- **Nudge Type:** Push Notification  
- **Content Template:** Template 6  

**Rule 7**  
- **Churn Score:** 0.7 – 0.9  
- **Churn Reason:** INACTIVITY  
- **Nudge Type:** Push Notification  
- **Content Template:** Template 7  

**Rule 8**  
- **Churn Score:** 0.6 – 0.8  
- **Churn Reason:** CART_ABANDONMENT, LOW_ENGAGEMENT  
- **Nudge Type:** Email + Discount Coupon  
- **Content Template:** Template 8  

**Rule 9**  
- **Churn Score:** 0.75 – 0.95  
- **Churn Reason:** DELIVERY_ISSUES, PRICE_SENSITIVITY  
- **Nudge Type:** Push Notification  
- **Content Template:** Template 9  

**Rule 10**  
- **Churn Score:** 0.8 – 1.0  
- **Churn Reason:** PAYMENT_FAILURE, CART_ABANDONMENT  
- **Nudge Type:** Push Notification + Discount Coupon + Email  
- **Content Template:** Template 10  

---

## APIs to Expose

1. **Feature Ingestion API** – accept user events, transactions, and engagement data.  
2. **Churn Prediction API** – fetch features from Aerospike FS and call the model to predict and return churn probability and reasons.  
3. **Nudge Trigger API** – send retention nudges based on churn score and reasons.  
4. **Monitoring API** – track API performance, feature freshness, model accuracy, and nudge responses.

---

## Model


* **Algorithm:** XGBoost  
* **Task:** Churn probability scoring (0.0 to 1.0)  
* **Input:** Feature vector from Aerospike FS  
* **Output:**  
  - `churn_probability`: float (0.0 = will retain, 1.0 = will churn)  
  - `risk_segment`: string ("low", "medium", "high", "critical")  
  - `churn_reasons`: list of strings (enum values)  
  - `confidence_score`: float  

### Model Training (Separate Pipeline)
- **Real-time Event-based Training:**  
  - Stream user events, transactions, and engagement data into a training feature store.  
  - Incrementally update the model with **recent data** to adapt to changing user behavior.  

- **Batch Training:**  
  - Periodically retrain the model on **historical aggregated data** to refresh feature importance and improve overall performance.  
  - Store trained models in a **model registry** with versioning.  

- **Decoupling:**  
  - Training is **fully separated** from prediction/nudge flow.  
  - Prediction microservice always loads the **latest stable model** from the registry.


---

## Best Practices

* Run the model in a separate Docker container for isolation and scalability.  
* create separete files fo each service and maintain right directory structure

## Success Metrics
- Nudge delivery rates by channel
- User response rates by nudge type
- Churn prediction accuracy over time  
- Revenue impact of retention campaigns



