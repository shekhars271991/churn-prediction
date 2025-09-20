"""
Churn Prediction Model Module
Extracted from model-service for easier debugging
"""
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import joblib
import os
from feature_config import (
    FEATURE_COLUMNS, FEATURE_MAPPING, CATEGORICAL_MAPPINGS, 
    CATEGORICAL_INDICES, MODEL_PARAMS, SYNTHETIC_DATA_PARAMS
)

# Configure logging
logger = logging.getLogger(__name__)

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a synthetic one for POC"""
        # Try multiple model paths and formats
        model_paths = [
            "churn_model.joblib",  # Current directory .joblib
            "churn_model.pkl",     # Current directory .pkl
            "../models/churn_model.joblib",  # Models directory .joblib
            "../models/churn_model.pkl",     # Models directory .pkl
            "/app/models/churn_model.joblib", # Absolute path .joblib
            "/app/models/churn_model.pkl"     # Absolute path .pkl
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = joblib.load(model_path)
                    # Set feature columns for loaded model
                    self._set_feature_columns()
                    logger.info(f"Loaded existing churn model from: {model_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_path}: {e}")
                    continue
        
        logger.info("No existing model found, creating new synthetic model")
        
        # Create synthetic model for POC
        self._create_synthetic_model()
        
        # Save the model
        try:
            joblib.dump(self.model, model_path)
            logger.info("Saved synthetic churn model")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _set_feature_columns(self):
        """Set feature columns for the model"""
        self.feature_columns = FEATURE_COLUMNS
    
    def _create_synthetic_model(self):
        """Create a synthetic XGBoost model for POC"""
        logger.info("Creating synthetic churn model for POC")
        
        # Set feature columns
        self._set_feature_columns()
        
        # Generate synthetic training data
        np.random.seed(SYNTHETIC_DATA_PARAMS['random_seed'])
        n_samples = SYNTHETIC_DATA_PARAMS['n_samples']
        
        # Create synthetic features
        X_synthetic = np.random.randn(n_samples, len(self.feature_columns))
        
        # Create synthetic labels (churn probability based on some features)
        y_synthetic = (
            (X_synthetic[:, 0] < -0.5) |  # Low account age
            (X_synthetic[:, 7] > 1.0) |   # High days since last login
            (X_synthetic[:, 13] > 1.0)    # High cart abandonment
        ).astype(int)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(**MODEL_PARAMS)
        
        self.model.fit(X_synthetic, y_synthetic)
        logger.info("Synthetic model training completed")
    
    def prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction"""
        # Create feature vector with defaults
        feature_vector = np.zeros(len(self.feature_columns))
        
        # Fill in available features
        for feature_name, value in features.items():
            if feature_name in FEATURE_MAPPING and value is not None:
                idx = FEATURE_MAPPING[feature_name]
                if isinstance(value, (int, float)):
                    feature_vector[idx] = float(value)
                elif isinstance(value, bool):
                    feature_vector[idx] = float(value)
        
        # Handle categorical features with simple encoding
        for cat_feature, mapping in CATEGORICAL_MAPPINGS.items():
            if cat_feature in features and features[cat_feature] is not None:
                encoded_value = mapping.get(features[cat_feature], 0)
                idx = CATEGORICAL_INDICES[cat_feature]
                feature_vector[idx] = float(encoded_value)
        
        return feature_vector.reshape(1, -1)
    
    def predict_churn(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict churn probability and generate insights"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Prepare features
        feature_vector = self.prepare_features(features)
        
        # Get prediction probability
        churn_probability = float(self.model.predict_proba(feature_vector)[0][1])
        
        # Determine risk segment
        if churn_probability >= 0.8:
            risk_segment = "critical"
        elif churn_probability >= 0.6:
            risk_segment = "high"
        elif churn_probability >= 0.4:
            risk_segment = "medium"
        else:
            risk_segment = "low"
        
        # Generate churn reasons based on features
        churn_reasons = self._generate_churn_reasons(features, churn_probability)
        
        # Calculate confidence score
        confidence_score = min(0.95, max(0.6, abs(churn_probability - 0.5) * 2))
        
        return {
            "churn_probability": churn_probability,
            "risk_segment": risk_segment,
            "churn_reasons": churn_reasons,
            "confidence_score": confidence_score
        }
    
    def _generate_churn_reasons(self, features: Dict[str, Any], churn_prob: float) -> List[str]:
        """Generate churn reasons based on feature analysis"""
        reasons = []
        
        # Check various risk factors
        if features.get('days_last_login', 0) > 7:
            reasons.append("INACTIVITY")
        
        if features.get('cart_abandon', 0) > 0.5:
            reasons.append("CART_ABANDONMENT")
        
        if features.get('sess_7d', 0) < 2:
            reasons.append("LOW_ENGAGEMENT")
        
        if features.get('csat_score', 5) < 3:
            reasons.append("POOR_SUPPORT_EXPERIENCE")
        
        if features.get('refund_rate', 0) > 0.3:
            reasons.append("HIGH_REFUND_RATE")
        
        if features.get('days_last_purch', 0) > 30:
            reasons.append("PURCHASE_DECLINE")
        
        if features.get('tickets_90d', 0) > 3:
            reasons.append("SUPPORT_ISSUES")
        
        # If no specific reasons but high churn probability, add generic reason
        if not reasons and churn_prob > 0.6:
            reasons.append("BEHAVIORAL_PATTERNS")
        
        return reasons[:3]  # Return top 3 reasons

# Global predictor instance
churn_predictor = ChurnPredictor()

def get_model_health() -> Dict[str, Any]:
    """Get model health status"""
    return {
        "model_loaded": churn_predictor.model is not None,
        "feature_count": len(churn_predictor.feature_columns) if churn_predictor.feature_columns else 0,
        "timestamp": datetime.utcnow().isoformat()
    }
