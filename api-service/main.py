from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import aerospike
import httpx
import os
import json
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from model_predictor import churn_predictor, get_model_health
from nudge_engine import nudge_engine, get_nudge_health, NudgeResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    connect_aerospike()
    yield
    # Shutdown (if needed)
    pass

app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

# Environment variables
AEROSPIKE_HOST = os.getenv("AEROSPIKE_HOST", "aerospike")
AEROSPIKE_PORT = int(os.getenv("AEROSPIKE_PORT", "3000"))
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")
# NUDGE_SERVICE_URL = os.getenv("NUDGE_SERVICE_URL", "http://localhost:8002")  # No longer needed - integrated

# Aerospike client (will be initialized on startup)
client = None

def connect_aerospike():
    """Connect to Aerospike with retry logic"""
    global client
    config = {
        'hosts': [(AEROSPIKE_HOST, AEROSPIKE_PORT)]
    }
    try:
        logger.info(f"Attempting to connect to Aerospike at {AEROSPIKE_HOST}:{AEROSPIKE_PORT}")
        client = aerospike.client(config).connect()
        logger.info("Connected to Aerospike successfully")
        # Test the connection
        info = client.info_all("build")
        logger.info(f"Aerospike info: {info}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Aerospike: {str(e)}")
        client = None
        return False

# Pydantic models
class UserProfileFeatures(BaseModel):
    user_id: str
    acc_age_days: Optional[int] = None
    member_dur: Optional[int] = None
    loyalty_tier: Optional[str] = None
    geo_location: Optional[str] = None
    device_type: Optional[str] = None
    pref_payment: Optional[str] = None
    lang_pref: Optional[str] = None

class UserBehaviorFeatures(BaseModel):
    user_id: str
    days_last_login: Optional[int] = None
    days_last_purch: Optional[int] = None
    sess_7d: Optional[int] = None
    sess_30d: Optional[int] = None
    avg_sess_dur: Optional[float] = None
    ctr_10_sess: Optional[float] = None
    cart_abandon: Optional[float] = None
    wishlist_ratio: Optional[float] = None
    content_engage: Optional[float] = None

class TransactionalFeatures(BaseModel):
    user_id: str
    avg_order_val: Optional[float] = None
    orders_6m: Optional[int] = None
    purch_freq_90d: Optional[float] = None
    last_hv_purch: Optional[int] = None
    refund_rate: Optional[float] = None
    sub_pay_status: Optional[str] = None
    discount_dep: Optional[float] = None
    cat_spend_dist: Optional[Dict[str, float]] = None

class EngagementFeatures(BaseModel):
    user_id: str
    push_open_rate: Optional[float] = None
    email_ctr: Optional[float] = None
    inapp_ctr: Optional[float] = None
    promo_resp_time: Optional[float] = None
    retention_resp: Optional[str] = None

class SupportFeatures(BaseModel):
    user_id: str
    tickets_90d: Optional[int] = None
    avg_ticket_res: Optional[float] = None
    csat_score: Optional[float] = None
    refund_req: Optional[int] = None

class RealTimeSessionFeatures(BaseModel):
    user_id: str
    curr_sess_clk: Optional[int] = None
    checkout_time: Optional[float] = None
    cart_no_buy: Optional[bool] = None
    bounce_flag: Optional[bool] = None

class ChurnPredictionResponse(BaseModel):
    user_id: str
    churn_probability: float
    risk_segment: str
    churn_reasons: List[str]
    confidence_score: float
    features_retrieved: Dict[str, Any]
    feature_freshness: str
    prediction_timestamp: str
    nudges_triggered: Optional[List[Dict[str, Any]]] = None
    nudge_rule_matched: Optional[str] = None

class MonitoringMetrics(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    api_performance: Dict[str, Any]
    feature_freshness: Dict[str, Any]
    model_accuracy: Dict[str, Any]
    nudge_responses: Dict[str, Any]

# Helper functions
def store_features_in_aerospike(user_id: str, features: Dict[str, Any], feature_type: str):
    """Store features in Aerospike with proper key structure"""
    global client
    
    # Try to reconnect if client is None
    if client is None:
        if not connect_aerospike():
            raise HTTPException(status_code=503, detail="Aerospike not available")
    
    try:
        namespace = "churn_features"
        set_name = "users"
        key_name = user_id + "_" + feature_type
        key = (namespace, set_name, key_name)
        features_with_timestamp = {
            **features,
            "timestamp": datetime.utcnow().isoformat(),
            "feature_type": feature_type
        }
        client.put(key, features_with_timestamp)
        logger.info(f"Stored {feature_type} features for user {user_id}")
    except Exception as e:
        logger.error(f"Error storing features for user {user_id}: {str(e)}")
        # Try to reconnect on error
        client = None
        raise HTTPException(status_code=500, detail=f"Failed to store features: {str(e)}")

def retrieve_all_features(user_id: str) -> Dict[str, Any]:
    """Retrieve all feature types for a user from Aerospike"""
    global client
    
    # Try to reconnect if client is None
    if client is None:
        if not connect_aerospike():
            raise HTTPException(status_code=503, detail="Aerospike not available")
    
    feature_types = ["profile", "behavior", "transactional", "engagement", "support", "realtime"]
    all_features = {}
    feature_freshness = None
    
    for feature_type in feature_types:
        try:
            namespace = "churn_features"
            set_name = "users"
            key_name = user_id + "_" + feature_type
            key = (namespace, set_name, key_name)
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
            # Try to reconnect on error
            client = None
    
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
        
        # Call local model for prediction
        prediction_data = churn_predictor.predict_churn(features)
        
        # Trigger nudges using integrated nudge engine
        nudge_response = None
        if prediction_data["risk_segment"] in ["high", "critical"]:
            try:
                nudge_response = nudge_engine.trigger_nudges(
                    user_id=user_id,
                    churn_probability=prediction_data["churn_probability"],
                    risk_segment=prediction_data["risk_segment"],
                    churn_reasons=prediction_data["churn_reasons"]
                )
            except Exception as e:
                logger.error(f"Failed to trigger nudge for user {user_id}: {str(e)}")
        
        # Prepare response
        response = ChurnPredictionResponse(
            user_id=user_id,
            churn_probability=prediction_data["churn_probability"],
            risk_segment=prediction_data["risk_segment"],
            churn_reasons=prediction_data["churn_reasons"],
            confidence_score=prediction_data["confidence_score"],
            features_retrieved=features,
            feature_freshness=feature_freshness,
            prediction_timestamp=datetime.utcnow().isoformat(),
            nudges_triggered=[nudge.dict() for nudge in nudge_response.nudges_triggered] if nudge_response else None,
            nudge_rule_matched=nudge_response.rule_matched if nudge_response else None
        )
        
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
    global client
    
    try:
        aerospike_status = "disconnected"
        if client is not None:
            try:
                # Test basic connection
                info = client.info_all("build")
                aerospike_status = "connected"
            except Exception as e:
                aerospike_status = f"error: {str(e)}"
                client = None
        
        # Get model health
        model_health = get_model_health()
        
        # Get nudge engine health
        nudge_health = get_nudge_health()
        
        return {
            "status": "healthy" if aerospike_status == "connected" and model_health["model_loaded"] else "degraded",
            "aerospike": aerospike_status,
            "model": model_health,
            "nudge_engine": nudge_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Nudge Engine Endpoints
@app.get("/nudge/rules")
async def get_nudge_rules():
    """Get all nudge rules"""
    return nudge_engine.get_rules()

@app.get("/nudge/rules/{rule_id}")
async def get_nudge_rule(rule_id: str):
    """Get specific nudge rule by ID"""
    rule = nudge_engine.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return rule

@app.get("/nudge/test/{user_id}")
async def test_nudge_rules(
    user_id: str, 
    churn_probability: float = Query(..., description="Churn probability (0.0-1.0)"),
    churn_reasons: List[str] = Query(..., description="List of churn reasons")
):
    """Test which rule would match for given parameters"""
    return nudge_engine.test_rules(user_id, churn_probability, churn_reasons)

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Default values
    host = "0.0.0.0"
    port = 8000
    
    # Parse command line arguments
    if "--host" in sys.argv:
        host_index = sys.argv.index("--host") + 1
        if host_index < len(sys.argv):
            host = sys.argv[host_index]
    
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            port = int(sys.argv[port_index])
    
    uvicorn.run(app, host=host, port=port)
