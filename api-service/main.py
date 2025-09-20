from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import aerospike
import httpx
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Environment variables
AEROSPIKE_HOST = os.getenv("AEROSPIKE_HOST", "localhost")
AEROSPIKE_PORT = int(os.getenv("AEROSPIKE_PORT", "3000"))
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")
NUDGE_SERVICE_URL = os.getenv("NUDGE_SERVICE_URL", "http://localhost:8002")

# Aerospike client
config = {
    'hosts': [(AEROSPIKE_HOST, AEROSPIKE_PORT)]
}
client = aerospike.client(config).connect()

# Pydantic models
class UserProfileFeatures(BaseModel):
    user_id: str
    account_age_days: Optional[int] = None
    membership_duration: Optional[int] = None
    loyalty_tier: Optional[str] = None
    geo_location: Optional[str] = None
    device_type: Optional[str] = None
    preferred_payment_method: Optional[str] = None
    language_preference: Optional[str] = None

class UserBehaviorFeatures(BaseModel):
    user_id: str
    days_since_last_login: Optional[int] = None
    days_since_last_purchase: Optional[int] = None
    sessions_last_7days: Optional[int] = None
    sessions_last_30days: Optional[int] = None
    avg_session_duration_last_30days: Optional[float] = None
    click_through_rate_last_10_sessions: Optional[float] = None
    cart_abandonment_rate: Optional[float] = None
    wishlist_adds_vs_purchases: Optional[float] = None
    content_engagement_rate: Optional[float] = None

class TransactionalFeatures(BaseModel):
    user_id: str
    avg_order_value: Optional[float] = None
    total_orders_last_6months: Optional[int] = None
    purchase_frequency_last_90days: Optional[float] = None
    time_since_last_high_value_purchase: Optional[int] = None
    refund_rate: Optional[float] = None
    subscription_payment_status: Optional[str] = None
    discount_dependency_score: Optional[float] = None
    category_spend_distribution: Optional[Dict[str, float]] = None

class EngagementFeatures(BaseModel):
    user_id: str
    push_notification_open_rate: Optional[float] = None
    email_click_rate: Optional[float] = None
    in_app_offer_click_rate: Optional[float] = None
    response_time_to_promotions: Optional[float] = None
    recent_retention_offer_response: Optional[str] = None

class SupportFeatures(BaseModel):
    user_id: str
    support_tickets_last_90days: Optional[int] = None
    avg_ticket_resolution_time: Optional[float] = None
    csat_score_last_interaction: Optional[float] = None
    refund_requests: Optional[int] = None

class RealTimeSessionFeatures(BaseModel):
    user_id: str
    current_session_clicks: Optional[int] = None
    time_spent_on_checkout_page: Optional[float] = None
    added_to_cart_but_not_bought_flag: Optional[bool] = None
    session_bounce_flag: Optional[bool] = None

class ChurnPredictionResponse(BaseModel):
    user_id: str
    churn_probability: float
    risk_segment: str
    churn_reasons: List[str]
    confidence_score: float
    features_retrieved: Dict[str, Any]
    feature_freshness: str
    prediction_timestamp: str

class MonitoringMetrics(BaseModel):
    api_performance: Dict[str, Any]
    feature_freshness: Dict[str, Any]
    model_accuracy: Dict[str, Any]
    nudge_responses: Dict[str, Any]

# Helper functions
def store_features_in_aerospike(user_id: str, features: Dict[str, Any], feature_type: str):
    """Store features in Aerospike with proper key structure"""
    try:
        key = (None, "churn_features", f"{user_id}_{feature_type}")
        features_with_timestamp = {
            **features,
            "timestamp": datetime.utcnow().isoformat(),
            "feature_type": feature_type
        }
        client.put(key, features_with_timestamp)
        logger.info(f"Stored {feature_type} features for user {user_id}")
    except Exception as e:
        logger.error(f"Error storing features for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store features: {str(e)}")

def retrieve_all_features(user_id: str) -> Dict[str, Any]:
    """Retrieve all feature types for a user from Aerospike"""
    feature_types = ["profile", "behavior", "transactional", "engagement", "support", "realtime"]
    all_features = {}
    feature_freshness = None
    
    for feature_type in feature_types:
        try:
            key = (None, "churn_features", f"{user_id}_{feature_type}")
            (key, metadata, bins) = client.get(key)
            if bins:
                # Remove metadata fields and merge features
                features = {k: v for k, v in bins.items() if k not in ["timestamp", "feature_type"]}
                all_features.update(features)
                if not feature_freshness or bins.get("timestamp", "") > feature_freshness:
                    feature_freshness = bins.get("timestamp")
        except aerospike.exception.RecordNotFound:
            logger.warning(f"No {feature_type} features found for user {user_id}")
        except Exception as e:
            logger.error(f"Error retrieving {feature_type} features for user {user_id}: {str(e)}")
    
    return all_features, feature_freshness or datetime.utcnow().isoformat()

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Churn Prediction API", "version": "1.0.0"}

@app.post("/ingest/profile")
async def ingest_profile_features(features: UserProfileFeatures):
    """Feature Ingestion API - User Profile Features"""
    feature_dict = features.dict(exclude_unset=True)
    user_id = feature_dict.pop("user_id")
    store_features_in_aerospike(user_id, feature_dict, "profile")
    return {"status": "success", "message": f"Profile features stored for user {user_id}"}

@app.post("/ingest/behavior")
async def ingest_behavior_features(features: UserBehaviorFeatures):
    """Feature Ingestion API - User Behavior Features"""
    feature_dict = features.dict(exclude_unset=True)
    user_id = feature_dict.pop("user_id")
    store_features_in_aerospike(user_id, feature_dict, "behavior")
    return {"status": "success", "message": f"Behavior features stored for user {user_id}"}

@app.post("/ingest/transactional")
async def ingest_transactional_features(features: TransactionalFeatures):
    """Feature Ingestion API - Transactional Features"""
    feature_dict = features.dict(exclude_unset=True)
    user_id = feature_dict.pop("user_id")
    store_features_in_aerospike(user_id, feature_dict, "transactional")
    return {"status": "success", "message": f"Transactional features stored for user {user_id}"}

@app.post("/ingest/engagement")
async def ingest_engagement_features(features: EngagementFeatures):
    """Feature Ingestion API - Engagement Features"""
    feature_dict = features.dict(exclude_unset=True)
    user_id = feature_dict.pop("user_id")
    store_features_in_aerospike(user_id, feature_dict, "engagement")
    return {"status": "success", "message": f"Engagement features stored for user {user_id}"}

@app.post("/ingest/support")
async def ingest_support_features(features: SupportFeatures):
    """Feature Ingestion API - Support Features"""
    feature_dict = features.dict(exclude_unset=True)
    user_id = feature_dict.pop("user_id")
    store_features_in_aerospike(user_id, feature_dict, "support")
    return {"status": "success", "message": f"Support features stored for user {user_id}"}

@app.post("/ingest/realtime")
async def ingest_realtime_features(features: RealTimeSessionFeatures):
    """Feature Ingestion API - Real-time Session Features"""
    feature_dict = features.dict(exclude_unset=True)
    user_id = feature_dict.pop("user_id")
    store_features_in_aerospike(user_id, feature_dict, "realtime")
    return {"status": "success", "message": f"Real-time features stored for user {user_id}"}

@app.post("/predict/{user_id}")
async def predict_churn(user_id: str) -> ChurnPredictionResponse:
    """Churn Prediction API - Fetch features and predict churn probability"""
    try:
        # Retrieve all features from Aerospike
        features, feature_freshness = retrieve_all_features(user_id)
        
        if not features:
            raise HTTPException(status_code=404, detail=f"No features found for user {user_id}")
        
        # Call model service for prediction
        async with httpx.AsyncClient() as client_http:
            model_response = await client_http.post(
                f"{MODEL_SERVICE_URL}/predict",
                json={"user_id": user_id, "features": features}
            )
            
            if model_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Model service error")
            
            prediction_data = model_response.json()
        
        # Prepare response
        response = ChurnPredictionResponse(
            user_id=user_id,
            churn_probability=prediction_data["churn_probability"],
            risk_segment=prediction_data["risk_segment"],
            churn_reasons=prediction_data["churn_reasons"],
            confidence_score=prediction_data["confidence_score"],
            features_retrieved=features,
            feature_freshness=feature_freshness,
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
        # Trigger nudge if high or critical risk
        if prediction_data["risk_segment"] in ["high", "critical"]:
            try:
                async with httpx.AsyncClient() as client_http:
                    await client_http.post(
                        f"{NUDGE_SERVICE_URL}/trigger",
                        json={
                            "user_id": user_id,
                            "churn_probability": prediction_data["churn_probability"],
                            "risk_segment": prediction_data["risk_segment"],
                            "churn_reasons": prediction_data["churn_reasons"]
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to trigger nudge for user {user_id}: {str(e)}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting churn for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/monitoring")
async def get_monitoring_metrics() -> MonitoringMetrics:
    """Monitoring API - Track API performance, feature freshness, and model accuracy"""
    # This is a placeholder implementation for POC
    # In production, this would collect real metrics from various sources
    return MonitoringMetrics(
        api_performance={
            "total_requests": 1000,
            "avg_response_time_ms": 150,
            "error_rate": 0.01
        },
        feature_freshness={
            "avg_feature_age_hours": 2.5,
            "stale_features_count": 5
        },
        model_accuracy={
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81
        },
        nudge_responses={
            "nudges_sent": 250,
            "response_rate": 0.15,
            "conversion_rate": 0.08
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Aerospike connection
        client.info_all()
        return {"status": "healthy", "aerospike": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
