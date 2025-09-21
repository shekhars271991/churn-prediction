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
- `GET /nudge/rules` - View all nudge rules
- `GET /nudge/rules/{rule_id}` - Get specific nudge rule
- `GET /nudge/test/{user_id}` - Test nudge rule matching
- `GET /health` - Health check

## 🏗️ Architecture

- **API Service**: Feature ingestion + churn prediction + nudge triggering (integrated model & nudge engine)
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
  "nudges_triggered": [{"type": "Push Notification", "channel": "push"}],
  "nudge_rule_matched": "rule_2"
}
```

## 🔧 Development

### Debug Mode
```bash
# Use VS Code debugger with .vscode/launch.json
# API Service: localhost:8100 (includes integrated nudge engine)
```

### Directory Structure
```
├── api-service/         # Main API + integrated model + nudge engine
├── training-service/    # Model training + data generation  
├── models/             # Trained model artifacts
└── docker-compose.yml  # Service orchestration
```

## 📈 Production Notes

This is a POC. For production: add authentication, monitoring, model versioning, and real nudge delivery systems.
