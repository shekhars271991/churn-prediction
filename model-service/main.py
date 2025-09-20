from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Model Service", version="1.0.0")

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")

# Global model variables
churn_model = None
feature_columns = None

# Pydantic models
class PredictionRequest(BaseModel):
    user_id: str
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    user_id: str
    churn_probability: float
    risk_segment: str
    churn_reasons: List[str]
    confidence_score: float

# Churn reasons enum as defined in the plan
CHURN_REASONS = [
    "INACTIVITY",
    "CART_ABANDONMENT", 
    "LOW_ENGAGEMENT",
    "PRICE_SENSITIVITY",
    "DELIVERY_ISSUES",
    "PRODUCT_AVAILABILITY",
    "PAYMENT_FAILURE"
]

# Feature columns as defined in the plan (POC subset)
FEATURE_COLUMNS = [
    "days_since_last_login",
    "sessions_last_7days",
    "avg_order_value", 
    "total_orders_last_6months",
    "account_age_days",
    "support_tickets_last_90days",
    "push_notification_open_rate",
    "cart_abandonment_rate",
    "avg_session_duration_last_30days",
    "purchase_frequency_last_90days"
]

def load_model():
    """Load the XGBoost model from file or create a dummy model for POC"""
    global churn_model, feature_columns
    
    model_file = os.path.join(MODEL_PATH, "churn_model.pkl")
    
    if os.path.exists(model_file):
        try:
            churn_model = joblib.load(model_file)
            logger.info("Loaded existing churn model")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            churn_model = create_dummy_model()
    else:
        logger.info("No existing model found, creating dummy model for POC")
        churn_model = create_dummy_model()
    
    feature_columns = FEATURE_COLUMNS

def create_dummy_model():
    """Create a dummy XGBoost model for POC purposes"""
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_features = len(FEATURE_COLUMNS)
    
    # Create synthetic feature data
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels with some logic
    # Higher churn probability for users with:
    # - High days_since_last_login
    # - Low sessions_last_7days  
    # - High support_tickets_last_90days
    churn_prob = (
        0.3 * (X[:, 0] > 0) +  # days_since_last_login
        0.2 * (X[:, 1] < 0) +  # sessions_last_7days (inverted)
        0.3 * (X[:, 5] > 0) +  # support_tickets_last_90days
        0.2 * np.random.random(n_samples)
    )
    y = (churn_prob > 0.5).astype(int)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    # Save the dummy model
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_PATH, "churn_model.pkl"))
    logger.info("Created and saved dummy XGBoost model")
    
    return model

def prepare_features(features: Dict[str, Any]) -> np.ndarray:
    """Prepare feature vector for model prediction"""
    feature_vector = []
    
    for col in feature_columns:
        value = features.get(col, 0)  # Default to 0 if feature missing
        
        # Handle None values
        if value is None:
            value = 0
            
        # Convert to float
        try:
            value = float(value)
        except (ValueError, TypeError):
            value = 0.0
            
        feature_vector.append(value)
    
    return np.array(feature_vector).reshape(1, -1)

def determine_risk_segment(probability: float) -> str:
    """Determine risk segment based on churn probability"""
    if probability >= 0.8:
        return "critical"
    elif probability >= 0.6:
        return "high"
    elif probability >= 0.3:
        return "medium"
    else:
        return "low"

def determine_churn_reasons(features: Dict[str, Any], probability: float) -> List[str]:
    """Determine churn reasons based on features and probability"""
    reasons = []
    
    # Rule-based churn reason detection
    days_since_login = features.get("days_since_last_login", 0) or 0
    sessions_7days = features.get("sessions_last_7days", 0) or 0
    cart_abandonment = features.get("cart_abandonment_rate", 0) or 0
    support_tickets = features.get("support_tickets_last_90days", 0) or 0
    avg_order_value = features.get("avg_order_value", 0) or 0
    push_open_rate = features.get("push_notification_open_rate", 0) or 0
    
    # Inactivity detection
    if days_since_login > 7 or sessions_7days < 2:
        reasons.append("INACTIVITY")
    
    # Cart abandonment
    if cart_abandonment > 0.5:
        reasons.append("CART_ABANDONMENT")
    
    # Low engagement
    if push_open_rate < 0.2 and sessions_7days < 3:
        reasons.append("LOW_ENGAGEMENT")
    
    # Price sensitivity (low AOV with high churn probability)
    if avg_order_value < 30 and probability > 0.6:
        reasons.append("PRICE_SENSITIVITY")
    
    # Support issues
    if support_tickets > 2:
        reasons.append("DELIVERY_ISSUES")
    
    # Default reason if none detected
    if not reasons and probability > 0.5:
        reasons.append("LOW_ENGAGEMENT")
    
    return reasons

def calculate_confidence_score(features: Dict[str, Any]) -> float:
    """Calculate confidence score based on feature completeness"""
    total_features = len(feature_columns)
    available_features = sum(1 for col in feature_columns if features.get(col) is not None)
    
    # Base confidence on feature completeness
    completeness_score = available_features / total_features
    
    # Add some randomness for POC
    confidence = min(0.95, completeness_score * 0.8 + np.random.random() * 0.2)
    
    return round(confidence, 2)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {"message": "Churn Model Service", "version": "1.0.0"}

@app.post("/predict")
async def predict_churn(request: PredictionRequest) -> PredictionResponse:
    """Predict churn probability and reasons for a user"""
    try:
        if churn_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Prepare features
        feature_vector = prepare_features(request.features)
        
        # Get prediction probability
        churn_probability = float(churn_model.predict_proba(feature_vector)[0][1])
        
        # Determine risk segment
        risk_segment = determine_risk_segment(churn_probability)
        
        # Determine churn reasons
        churn_reasons = determine_churn_reasons(request.features, churn_probability)
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(request.features)
        
        logger.info(f"Predicted churn for user {request.user_id}: {churn_probability:.3f} ({risk_segment})")
        
        return PredictionResponse(
            user_id=request.user_id,
            churn_probability=round(churn_probability, 3),
            risk_segment=risk_segment,
            churn_reasons=churn_reasons,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.error(f"Error predicting churn for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if churn_model is None:
        return {"status": "no_model_loaded"}
    
    return {
        "model_type": "XGBoost",
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "model_loaded": True,
        "churn_reasons": CHURN_REASONS
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": churn_model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Default values
    host = "0.0.0.0"
    port = 8001
    
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
