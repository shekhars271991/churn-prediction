# Churn Prediction Microservice

A Python microservice for churn prediction using Aerospike Feature Store and XGBoost, built according to the detailed plan in `plan.md`.

## Architecture

- **API Service** (Port 8000): Feature ingestion and churn prediction APIs
- **Model Service** (Port 8001): XGBoost model for churn scoring and reason detection  
- **Nudge Service** (Port 8002): Rule-based nudge triggering system
- **Training Service**: Batch model training pipeline
- **Aerospike**: Feature store for real-time feature storage and retrieval

## Quick Start

### 1. Start the Services

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8000/health
curl http://localhost:8001/health  
curl http://localhost:8002/health
```

### 2. Train the Model (Optional)

```bash
# Train a new model with synthetic data
docker-compose --profile training up training-service
```

### 3. Generate Synthetic Data

```bash
# Install dependencies
cd data
pip install -r requirements.txt

# Generate synthetic user data and test the system
python generate_synthetic_data.py
```

### 4. Test the APIs

```bash
# Ingest user features
curl -X POST "http://localhost:8000/ingest/profile" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_001",
    "account_age_days": 365,
    "loyalty_tier": "gold",
    "geo_location": "US-CA"
  }'

# Get churn prediction
curl -X POST "http://localhost:8000/predict/test_user_001"
```

## API Endpoints

### API Service (Port 8000)

- `POST /ingest/profile` - Ingest user profile features
- `POST /ingest/behavior` - Ingest user behavior features  
- `POST /ingest/transactional` - Ingest transactional features
- `POST /ingest/engagement` - Ingest engagement features
- `POST /ingest/support` - Ingest support features
- `POST /ingest/realtime` - Ingest real-time session features
- `POST /predict/{user_id}` - Get churn prediction with automatic nudge triggering
- `GET /monitoring` - Get system monitoring metrics
- `GET /health` - Health check

### Model Service (Port 8001)

- `POST /predict` - Get churn probability and reasons
- `GET /model/info` - Get model information
- `GET /health` - Health check

### Nudge Service (Port 8002)

- `POST /trigger` - Trigger nudges based on churn score and reasons
- `GET /rules` - Get all nudge rules
- `GET /rules/{rule_id}` - Get specific nudge rule
- `POST /test/{user_id}` - Test which rule would match given parameters
- `GET /health` - Health check

## Features

The system implements all features from the plan:

### User Profile Features
- user_id, account_age_days, membership_duration, loyalty_tier, geo_location, device_type, preferred_payment_method, language_preference

### User Behavior Features  
- days_since_last_login, sessions_last_7days/30days, avg_session_duration, click_through_rate, cart_abandonment_rate, etc.

### Transactional Features
- avg_order_value, total_orders_last_6months, purchase_frequency, refund_rate, subscription_payment_status, etc.

### Engagement Features
- push_notification_open_rate, email_click_rate, in_app_offer_click_rate, response_time_to_promotions, etc.

### Support Features
- support_tickets_last_90days, avg_ticket_resolution_time, csat_score_last_interaction, refund_requests

### Real-time Session Features
- current_session_clicks, time_spent_on_checkout_page, added_to_cart_but_not_bought_flag, session_bounce_flag

## Nudge Rules

The system implements all 10 nudge rules from the plan, mapping churn scores and reasons to appropriate nudge types:

- **Rule 1-10**: Various combinations of churn score ranges (0.6-1.0) and churn reasons (INACTIVITY, CART_ABANDONMENT, LOW_ENGAGEMENT, PRICE_SENSITIVITY, DELIVERY_ISSUES, PRODUCT_AVAILABILITY, PAYMENT_FAILURE)
- **Nudge Types**: Email, Push Notification, Discount Coupon
- **Content Templates**: Template 1-10 for different scenarios

## Churn Reasons

The model detects these churn reasons:
- INACTIVITY
- CART_ABANDONMENT  
- LOW_ENGAGEMENT
- PRICE_SENSITIVITY
- DELIVERY_ISSUES
- PRODUCT_AVAILABILITY
- PAYMENT_FAILURE

## Model Output

For each prediction request, the system returns:
- `churn_probability`: Float (0.0 to 1.0)
- `risk_segment`: String ("low", "medium", "high", "critical")  
- `churn_reasons`: List of detected churn reasons
- `confidence_score`: Model confidence in the prediction
- `features_retrieved`: All features used for prediction
- `feature_freshness`: Timestamp of most recent feature update

## Development

### Directory Structure

```
├── api-service/          # FastAPI service for feature ingestion and prediction
├── model-service/        # XGBoost model service  
├── nudge-service/        # Rule-based nudge triggering
├── training-service/     # Model training pipeline
├── data/                 # Synthetic data generation
├── models/               # Trained model artifacts
├── aerospike.conf        # Aerospike configuration
├── docker-compose.yml    # Service orchestration
└── plan.md              # Detailed system plan
```

### Adding New Features

1. Update feature schemas in `api-service/main.py`
2. Add feature columns to `FEATURE_COLUMNS` in `model-service/main.py`
3. Update training data generation in `training-service/train.py`
4. Retrain model with new features

### Adding New Nudge Rules

1. Add rule to `NUDGE_RULES` in `nudge-service/main.py`
2. Update rule priority and conflict resolution logic
3. Test with `/test/{user_id}` endpoint

## Monitoring

The system provides monitoring through:
- API performance metrics
- Feature freshness tracking  
- Model accuracy monitoring
- Nudge response rates

Access monitoring data via `GET /monitoring` endpoint.

## Production Considerations

This is a POC implementation. For production:

1. **Security**: Add JWT authentication, API rate limiting
2. **Scalability**: Use Kubernetes, add load balancers
3. **Monitoring**: Integrate with Prometheus/Grafana
4. **Data Pipeline**: Use Kafka/Pulsar for real-time ingestion
5. **Model Management**: Add model versioning and A/B testing
6. **Nudge Delivery**: Integrate with actual email/push services
