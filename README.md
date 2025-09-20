# Churn Prediction Microservice

A Python microservice for churn prediction using Aerospike Feature Store and XGBoost.

## 🚀 Quick Start

### 1. Start Services
```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### 2. Train Model (Generates 5000 synthetic users)
```bash
docker-compose --profile training up training-service
```

### 3. Test Prediction
```bash
# Ingest features
curl -X POST "http://localhost:8000/ingest/profile" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_001", "acc_age_days": 365, "loyalty_tier": "gold"}'

# Get prediction
curl -X POST "http://localhost:8000/predict/test_001"
```

## 📡 API Endpoints

### API Service (Port 8000)
- `POST /ingest/{feature_type}` - Ingest features (profile, behavior, transactional, engagement, support, realtime)
- `POST /predict/{user_id}` - Get churn prediction + auto-trigger nudges
- `GET /health` - Health check

### Nudge Service (Port 8002)  
- `POST /trigger` - Trigger nudges based on churn score
- `GET /rules` - View nudge rules
- `GET /health` - Health check

## 🏗️ Architecture

- **API Service**: Feature ingestion + churn prediction (integrated model)
- **Nudge Service**: Rule-based nudge triggering  
- **Training Service**: Synthetic data generation + model training
- **Aerospike**: Real-time feature store

## 🎯 Features

**Profile**: Account age, loyalty tier, geo location, device type  
**Behavior**: Login patterns, session data, cart abandonment  
**Transactional**: Order value, purchase frequency, refunds  
**Engagement**: Push/email rates, promo responses  
**Support**: Tickets, CSAT scores, resolution times  
**Real-time**: Session clicks, checkout time, bounce flags

## 📊 Model Output

```json
{
  "churn_probability": 0.75,
  "risk_segment": "high", 
  "churn_reasons": ["INACTIVITY", "CART_ABANDONMENT"],
  "confidence_score": 0.89,
  "nudges_triggered": [...]
}
```
