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

# Configure logging
logger = logging.getLogger(__name__)

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a synthetic one for POC"""
        model_path = "churn_model.joblib"
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                # Set feature columns for loaded model
                self._set_feature_columns()
                logger.info("Loaded existing churn model")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, creating new one")
        
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
        self.feature_columns = [
            'acc_age_days', 'member_dur', 'loyalty_tier_encoded', 'geo_location_encoded',
            'device_type_encoded', 'pref_payment_encoded', 'lang_pref_encoded',
            'days_last_login', 'days_last_purch', 'sess_7d', 'sess_30d', 'avg_sess_dur',
            'ctr_10_sess', 'cart_abandon', 'wishlist_ratio', 'content_engage',
            'avg_order_val', 'orders_6m', 'purch_freq_90d', 'last_hv_purch', 'refund_rate',
            'sub_pay_status_encoded', 'discount_dep', 'push_open_rate', 'email_ctr',
            'inapp_ctr', 'promo_resp_time', 'retention_resp_encoded', 'tickets_90d',
            'avg_ticket_res', 'csat_score', 'refund_req', 'curr_sess_clk', 'checkout_time',
            'cart_no_buy', 'bounce_flag'
        ]
    
    def _create_synthetic_model(self):
        """Create a synthetic XGBoost model for POC"""
        logger.info("Creating synthetic churn model for POC")
        
        # Set feature columns
        self._set_feature_columns()
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic features
        X_synthetic = np.random.randn(n_samples, len(self.feature_columns))
        
        # Create synthetic labels (churn probability based on some features)
        y_synthetic = (
            (X_synthetic[:, 0] < -0.5) |  # Low account age
            (X_synthetic[:, 7] > 1.0) |   # High days since last login
            (X_synthetic[:, 13] > 1.0)    # High cart abandonment
        ).astype(int)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_synthetic, y_synthetic)
        logger.info("Synthetic model training completed")
    
    def prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction"""
        # Create feature vector with defaults
        feature_vector = np.zeros(len(self.feature_columns))
        
        # Mapping from input features to model features
        feature_mapping = {
            'acc_age_days': 0, 'member_dur': 1, 'days_last_login': 7, 'days_last_purch': 8,
            'sess_7d': 9, 'sess_30d': 10, 'avg_sess_dur': 11, 'ctr_10_sess': 12,
            'cart_abandon': 13, 'wishlist_ratio': 14, 'content_engage': 15,
            'avg_order_val': 16, 'orders_6m': 17, 'purch_freq_90d': 18, 'last_hv_purch': 19,
            'refund_rate': 20, 'discount_dep': 22, 'push_open_rate': 23, 'email_ctr': 24,
            'inapp_ctr': 25, 'promo_resp_time': 26, 'tickets_90d': 28, 'avg_ticket_res': 29,
            'csat_score': 30, 'refund_req': 31, 'curr_sess_clk': 32, 'checkout_time': 33,
            'cart_no_buy': 34, 'bounce_flag': 35
        }
        
        # Fill in available features
        for feature_name, value in features.items():
            if feature_name in feature_mapping and value is not None:
                idx = feature_mapping[feature_name]
                if isinstance(value, (int, float)):
                    feature_vector[idx] = float(value)
                elif isinstance(value, bool):
                    feature_vector[idx] = float(value)
        
        # Handle categorical features with simple encoding
        categorical_mappings = {
            'loyalty_tier': {'bronze': 1, 'silver': 2, 'gold': 3, 'platinum': 4},
            'geo_location': {'US-CA': 1, 'US-NY': 2, 'US-TX': 3, 'UK': 4, 'DE': 5},
            'device_type': {'mobile': 1, 'desktop': 2, 'tablet': 3},
            'pref_payment': {'credit': 1, 'debit': 2, 'paypal': 3, 'crypto': 4},
            'lang_pref': {'en': 1, 'es': 2, 'fr': 3, 'de': 4},
            'sub_pay_status': {'active': 1, 'inactive': 2, 'cancelled': 3},
            'retention_resp': {'positive': 1, 'negative': 2, 'neutral': 3}
        }
        
        categorical_indices = {
            'loyalty_tier': 2, 'geo_location': 3, 'device_type': 4,
            'pref_payment': 5, 'lang_pref': 6, 'sub_pay_status': 21, 'retention_resp': 27
        }
        
        for cat_feature, mapping in categorical_mappings.items():
            if cat_feature in features and features[cat_feature] is not None:
                encoded_value = mapping.get(features[cat_feature], 0)
                idx = categorical_indices[cat_feature]
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
