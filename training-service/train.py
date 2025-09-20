"""
Comprehensive Training Service for Churn Prediction Model
1. Generates synthetic users with fake names
2. Creates realistic feature data for each user
3. Saves all data to Aerospike
4. Trains the model on this data
5. Exits cleanly
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import aerospike
import logging
from datetime import datetime, timedelta
import json
import random
from faker import Faker
from typing import Dict, List, Any
from feature_config import (
    FEATURE_COLUMNS, FEATURE_MAPPING, CATEGORICAL_MAPPINGS, 
    CATEGORICAL_INDICES, MODEL_PARAMS, SYNTHETIC_DATA_PARAMS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
NUM_USERS = int(os.getenv("NUM_USERS", "5000"))
NATIONALITY = os.getenv("NATIONALITY", "en_US")  # en_US, en_GB, de_DE, fr_FR, es_ES, etc.
AEROSPIKE_HOST = os.getenv("AEROSPIKE_HOST", "localhost")
AEROSPIKE_PORT = int(os.getenv("AEROSPIKE_PORT", "3000"))
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH", "churn_model.joblib")
CHURN_RATE = float(os.getenv("CHURN_RATE", "0.25"))  # 25% churn rate

class ChurnDataGenerator:
    def __init__(self, nationality: str = "en_US"):
        self.fake = Faker(nationality)
        Faker.seed(42)  # For reproducible results
        random.seed(42)
        np.random.seed(42)
        
    def generate_users(self, num_users: int) -> List[Dict[str, Any]]:
        """Generate synthetic users with realistic profiles"""
        logger.info(f"Generating {num_users} synthetic users...")
        
        users = []
        for i in range(num_users):
            user_id = f"user_{i+1:05d}"
            
            # Generate user profile
            user = {
                "user_id": user_id,
                "name": self.fake.name(),
                "email": self.fake.email(),
                "created_at": self.fake.date_between(start_date="-3y", end_date="today")
            }
            
            # Generate all feature types for this user
            user.update(self._generate_profile_features(user))
            user.update(self._generate_behavior_features(user))
            user.update(self._generate_transactional_features(user))
            user.update(self._generate_engagement_features(user))
            user.update(self._generate_support_features(user))
            user.update(self._generate_realtime_features(user))
            
            # Generate churn label based on features (realistic correlation)
            user["is_churned"] = self._determine_churn_label(user)
            
            users.append(user)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1} users...")
        
        logger.info(f"Generated {len(users)} users successfully")
        return users
    
    def _generate_profile_features(self, user: Dict) -> Dict[str, Any]:
        """Generate user profile features"""
        account_age = (datetime.now().date() - user["created_at"]).days
        
        return {
            "acc_age_days": account_age,
            "member_dur": max(1, account_age - random.randint(0, 30)),
            "loyalty_tier": np.random.choice(["bronze", "silver", "gold", "platinum"], 
                                           p=[0.4, 0.3, 0.2, 0.1]),
            "geo_location": np.random.choice(["US-CA", "US-NY", "US-TX", "UK", "DE"], 
                                           p=[0.25, 0.20, 0.15, 0.25, 0.15]),
            "device_type": np.random.choice(["mobile", "desktop", "tablet"], 
                                          p=[0.6, 0.3, 0.1]),
            "pref_payment": np.random.choice(["credit", "debit", "paypal", "crypto"], 
                                           p=[0.4, 0.3, 0.25, 0.05]),
            "lang_pref": np.random.choice(["en", "es", "fr", "de"], 
                                        p=[0.7, 0.15, 0.1, 0.05])
        }
    
    def _generate_behavior_features(self, user: Dict) -> Dict[str, Any]:
        """Generate user behavior features"""
        # Correlate with loyalty tier
        loyalty_multiplier = {"bronze": 0.5, "silver": 0.7, "gold": 1.0, "platinum": 1.5}
        multiplier = loyalty_multiplier.get(user.get("loyalty_tier", "bronze"), 0.5)
        
        return {
            "days_last_login": max(0, int(np.random.exponential(3) / multiplier)),
            "days_last_purch": max(0, int(np.random.exponential(7) / multiplier)),
            "sess_7d": max(0, int(np.random.poisson(5 * multiplier))),
            "sess_30d": max(0, int(np.random.poisson(20 * multiplier))),
            "avg_sess_dur": max(1.0, np.random.normal(15 * multiplier, 5)),
            "ctr_10_sess": min(1.0, max(0.0, np.random.beta(2, 5) * multiplier)),
            "cart_abandon": min(1.0, max(0.0, np.random.beta(2, 3) / multiplier)),
            "wishlist_ratio": min(5.0, max(0.0, np.random.gamma(2, 0.5) * multiplier)),
            "content_engage": min(1.0, max(0.0, np.random.beta(3, 2) * multiplier))
        }
    
    def _generate_transactional_features(self, user: Dict) -> Dict[str, Any]:
        """Generate transactional features"""
        loyalty_multiplier = {"bronze": 0.6, "silver": 0.8, "gold": 1.2, "platinum": 2.0}
        multiplier = loyalty_multiplier.get(user.get("loyalty_tier", "bronze"), 0.6)
        
        return {
            "avg_order_val": max(10.0, np.random.lognormal(4, 0.5) * multiplier),
            "orders_6m": max(0, int(np.random.poisson(8 * multiplier))),
            "purch_freq_90d": max(0.0, np.random.gamma(2, 1) * multiplier),
            "last_hv_purch": max(0, int(np.random.exponential(30) / multiplier)),
            "refund_rate": min(1.0, max(0.0, np.random.beta(1, 9) / multiplier)),
            "sub_pay_status": np.random.choice(["active", "inactive", "cancelled"], 
                                             p=[0.7, 0.2, 0.1]),
            "discount_dep": min(1.0, max(0.0, np.random.beta(2, 3) / multiplier))
        }
    
    def _generate_engagement_features(self, user: Dict) -> Dict[str, Any]:
        """Generate engagement features"""
        device_multiplier = {"mobile": 1.2, "desktop": 0.8, "tablet": 1.0}
        multiplier = device_multiplier.get(user.get("device_type", "mobile"), 1.0)
        
        return {
            "push_open_rate": min(1.0, max(0.0, np.random.beta(3, 2) * multiplier)),
            "email_ctr": min(1.0, max(0.0, np.random.beta(2, 8) * multiplier)),
            "inapp_ctr": min(1.0, max(0.0, np.random.beta(2, 5) * multiplier)),
            "promo_resp_time": max(0.1, np.random.exponential(2) / multiplier),
            "retention_resp": np.random.choice(["positive", "negative", "neutral"], 
                                             p=[0.3, 0.2, 0.5])
        }
    
    def _generate_support_features(self, user: Dict) -> Dict[str, Any]:
        """Generate support interaction features"""
        return {
            "tickets_90d": max(0, int(np.random.poisson(1.5))),
            "avg_ticket_res": max(0.5, np.random.lognormal(2, 0.5)),
            "csat_score": min(5.0, max(1.0, np.random.normal(4.2, 0.8))),
            "refund_req": max(0, int(np.random.poisson(0.5)))
        }
    
    def _generate_realtime_features(self, user: Dict) -> Dict[str, Any]:
        """Generate real-time session features"""
        return {
            "curr_sess_clk": max(0, int(np.random.poisson(8))),
            "checkout_time": max(0.0, np.random.exponential(3)),
            "cart_no_buy": bool(np.random.choice([True, False], p=[0.3, 0.7])),
            "bounce_flag": bool(np.random.choice([True, False], p=[0.2, 0.8]))
        }
    
    def _determine_churn_label(self, user: Dict) -> bool:
        """Determine churn label based on realistic feature correlations"""
        churn_score = 0.0
        
        # High-risk factors (increase churn probability)
        if user.get("days_last_login", 0) > 14:
            churn_score += 0.3
        if user.get("days_last_purch", 0) > 60:
            churn_score += 0.25
        if user.get("cart_abandon", 0) > 0.7:
            churn_score += 0.2
        if user.get("sess_7d", 0) < 2:
            churn_score += 0.2
        if user.get("csat_score", 5) < 3:
            churn_score += 0.15
        if user.get("refund_rate", 0) > 0.3:
            churn_score += 0.15
        if user.get("tickets_90d", 0) > 3:
            churn_score += 0.1
        
        # Protective factors (decrease churn probability)
        if user.get("loyalty_tier") in ["gold", "platinum"]:
            churn_score -= 0.2
        if user.get("orders_6m", 0) > 10:
            churn_score -= 0.15
        if user.get("push_open_rate", 0) > 0.5:
            churn_score -= 0.1
        
        # Convert to probability and make decision
        churn_probability = max(0.0, min(1.0, churn_score))
        return np.random.random() < churn_probability

class AerospikeDataManager:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = None
        self.connect()
    
    def connect(self):
        """Connect to Aerospike"""
        try:
            config = {'hosts': [(self.host, self.port)]}
            self.client = aerospike.client(config).connect()
            logger.info(f"Connected to Aerospike at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Aerospike: {e}")
            raise
    
    def save_user_features(self, users: List[Dict[str, Any]]):
        """Save user features to Aerospike by feature type"""
        logger.info(f"Saving features for {len(users)} users to Aerospike...")
        
        feature_types = {
            "profile": ["acc_age_days", "member_dur", "loyalty_tier", "geo_location", 
                       "device_type", "pref_payment", "lang_pref"],
            "behavior": ["days_last_login", "days_last_purch", "sess_7d", "sess_30d", 
                        "avg_sess_dur", "ctr_10_sess", "cart_abandon", "wishlist_ratio", "content_engage"],
            "transactional": ["avg_order_val", "orders_6m", "purch_freq_90d", "last_hv_purch", 
                             "refund_rate", "sub_pay_status", "discount_dep"],
            "engagement": ["push_open_rate", "email_ctr", "inapp_ctr", "promo_resp_time", "retention_resp"],
            "support": ["tickets_90d", "avg_ticket_res", "csat_score", "refund_req"],
            "realtime": ["curr_sess_clk", "checkout_time", "cart_no_buy", "bounce_flag"]
        }
        
        saved_count = 0
        for user in users:
            user_id = user["user_id"]
            
            for feature_type, feature_names in feature_types.items():
                # Extract features for this type
                features = {name: user.get(name) for name in feature_names if name in user}
                
                # Add metadata
                features_with_metadata = {
                    **features,
                    "timestamp": datetime.utcnow().isoformat(),
                    "feature_type": feature_type
                }
                
                # Convert data types for Aerospike compatibility
                aerospike_compatible_features = {}
                for k, v in features_with_metadata.items():
                    if isinstance(v, bool):
                        aerospike_compatible_features[k] = int(v)  # Convert bool to int
                    elif isinstance(v, np.bool_):
                        aerospike_compatible_features[k] = int(v)  # Convert numpy bool to int
                    elif isinstance(v, (np.integer, np.floating)):
                        aerospike_compatible_features[k] = float(v)  # Convert numpy numbers
                    elif v is None:
                        aerospike_compatible_features[k] = 0  # Convert None to 0
                    else:
                        aerospike_compatible_features[k] = v
                
                # Save to Aerospike
                try:
                    key = ("churn_features", "users", f"{user_id}_{feature_type}")
                    self.client.put(key, aerospike_compatible_features)
                except Exception as e:
                    logger.error(f"Failed to save {feature_type} features for {user_id}: {e}")
            
            saved_count += 1
            if saved_count % 1000 == 0:
                logger.info(f"Saved features for {saved_count} users...")
        
        logger.info(f"Successfully saved features for {saved_count} users")
    
    def close(self):
        """Close Aerospike connection"""
        if self.client:
            self.client.close()
            logger.info("Closed Aerospike connection")

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    def prepare_training_data(self, users: List[Dict[str, Any]]) -> tuple:
        """Prepare training data from user records"""
        logger.info("Preparing training data...")
        
        X = []
        y = []
        
        for user in users:
            # Create feature vector
            feature_vector = np.zeros(len(FEATURE_COLUMNS))
            
            # Fill numerical features
            for feature_name, value in user.items():
                if feature_name in FEATURE_MAPPING and value is not None:
                    idx = FEATURE_MAPPING[feature_name]
                    if isinstance(value, (int, float)):
                        feature_vector[idx] = float(value)
                    elif isinstance(value, bool):
                        feature_vector[idx] = float(value)
            
            # Handle categorical features
            for cat_feature, mapping in CATEGORICAL_MAPPINGS.items():
                if cat_feature in user and user[cat_feature] is not None:
                    encoded_value = mapping.get(user[cat_feature], 0)
                    idx = CATEGORICAL_INDICES[cat_feature]
                    feature_vector[idx] = float(encoded_value)
            
            X.append(feature_vector)
            y.append(int(user.get("is_churned", False)))
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Churn rate: {y.mean():.2%}")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the XGBoost model"""
        logger.info("Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = xgb.XGBClassifier(**MODEL_PARAMS)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            "auc_score": auc_score,
            "accuracy": class_report["accuracy"],
            "precision": class_report["1"]["precision"],
            "recall": class_report["1"]["recall"],
            "f1_score": class_report["1"]["f1-score"],
            "confusion_matrix": conf_matrix.tolist(),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        logger.info(f"Model training completed:")
        logger.info(f"  AUC Score: {auc_score:.4f}")
        logger.info(f"  Accuracy: {class_report['accuracy']:.4f}")
        logger.info(f"  Precision: {class_report['1']['precision']:.4f}")
        logger.info(f"  Recall: {class_report['1']['recall']:.4f}")
        logger.info(f"  F1 Score: {class_report['1']['f1-score']:.4f}")
        
        return metrics
    
    def save_model(self, output_path: str, metrics: Dict[str, Any]):
        """Save the trained model"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            
            # Save model
            joblib.dump(self.model, output_path)
            
            # Save metrics
            metrics_path = output_path.replace('.joblib', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Model saved to: {output_path}")
            logger.info(f"Metrics saved to: {metrics_path}")
            
            # Also save in .pkl format for compatibility with existing code
            pkl_path = output_path.replace('.joblib', '.pkl')
            joblib.dump(self.model, pkl_path)
            logger.info(f"Model also saved in .pkl format to: {pkl_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

def main():
    """Main training pipeline"""
    logger.info("Starting Churn Prediction Model Training Pipeline")
    logger.info(f"Configuration:")
    logger.info(f"  Users to generate: {NUM_USERS}")
    logger.info(f"  Nationality: {NATIONALITY}")
    logger.info(f"  Target churn rate: {CHURN_RATE:.1%}")
    logger.info(f"  Aerospike: {AEROSPIKE_HOST}:{AEROSPIKE_PORT}")
    
    try:
        # Step 1: Generate synthetic users and data
        logger.info("=" * 50)
        logger.info("STEP 1: Generating Synthetic Users")
        logger.info("=" * 50)
        
        data_generator = ChurnDataGenerator(nationality=NATIONALITY)
        users = data_generator.generate_users(NUM_USERS)
        
        # Step 2: Save data to Aerospike
        logger.info("=" * 50)
        logger.info("STEP 2: Saving Data to Aerospike")
        logger.info("=" * 50)
        
        aerospike_manager = AerospikeDataManager(AEROSPIKE_HOST, AEROSPIKE_PORT)
        aerospike_manager.save_user_features(users)
        
        # Step 3: Train model
        logger.info("=" * 50)
        logger.info("STEP 3: Training Model")
        logger.info("=" * 50)
        
        trainer = ModelTrainer()
        X, y = trainer.prepare_training_data(users)
        metrics = trainer.train_model(X, y)
        trainer.save_model(MODEL_OUTPUT_PATH, metrics)
        
        # Step 4: Cleanup
        logger.info("=" * 50)
        logger.info("STEP 4: Cleanup")
        logger.info("=" * 50)
        
        aerospike_manager.close()
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Generated {NUM_USERS} users with realistic churn patterns")
        logger.info(f"Model saved with AUC score: {metrics['auc_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
