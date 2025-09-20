import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import aerospike
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
AEROSPIKE_HOST = os.getenv("AEROSPIKE_HOST", "localhost")
AEROSPIKE_PORT = int(os.getenv("AEROSPIKE_PORT", "3000"))
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")
DATA_PATH = os.getenv("DATA_PATH", "/app/data")

# Feature columns as defined in the plan
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

def connect_to_aerospike():
    """Connect to Aerospike"""
    try:
        config = {'hosts': [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
        client = aerospike.client(config).connect()
        logger.info("Connected to Aerospike")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Aerospike: {str(e)}")
        return None

def generate_synthetic_training_data(n_samples=5000):
    """Generate synthetic training data for POC"""
    logger.info(f"Generating {n_samples} synthetic training samples")
    
    np.random.seed(42)
    
    # Generate realistic feature distributions
    data = {}
    
    # Days since last login (0-30 days, higher values indicate higher churn risk)
    data["days_since_last_login"] = np.random.exponential(5, n_samples).astype(int)
    data["days_since_last_login"] = np.clip(data["days_since_last_login"], 0, 30)
    
    # Sessions last 7 days (0-20 sessions, lower values indicate higher churn risk)
    data["sessions_last_7days"] = np.random.poisson(7, n_samples)
    data["sessions_last_7days"] = np.clip(data["sessions_last_7days"], 0, 20)
    
    # Average order value ($10-$500)
    data["avg_order_value"] = np.random.lognormal(3.5, 0.8, n_samples)
    data["avg_order_value"] = np.clip(data["avg_order_value"], 10, 500)
    
    # Total orders last 6 months (0-50)
    data["total_orders_last_6months"] = np.random.poisson(8, n_samples)
    data["total_orders_last_6months"] = np.clip(data["total_orders_last_6months"], 0, 50)
    
    # Account age in days (30-1000 days)
    data["account_age_days"] = np.random.uniform(30, 1000, n_samples).astype(int)
    
    # Support tickets last 90 days (0-10, higher values indicate issues)
    data["support_tickets_last_90days"] = np.random.poisson(1, n_samples)
    data["support_tickets_last_90days"] = np.clip(data["support_tickets_last_90days"], 0, 10)
    
    # Push notification open rate (0-1)
    data["push_notification_open_rate"] = np.random.beta(2, 3, n_samples)
    
    # Cart abandonment rate (0-1)
    data["cart_abandonment_rate"] = np.random.beta(2, 5, n_samples)
    
    # Average session duration last 30 days (1-60 minutes)
    data["avg_session_duration_last_30days"] = np.random.lognormal(2.5, 0.5, n_samples)
    data["avg_session_duration_last_30days"] = np.clip(data["avg_session_duration_last_30days"], 1, 60)
    
    # Purchase frequency last 90 days (0-1, purchases per day)
    data["purchase_frequency_last_90days"] = np.random.beta(1, 10, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate churn labels based on realistic business logic
    churn_probability = (
        0.15 * (df["days_since_last_login"] > 10) +  # Inactive users
        0.20 * (df["sessions_last_7days"] < 3) +     # Low engagement
        0.15 * (df["support_tickets_last_90days"] > 2) +  # Support issues
        0.10 * (df["cart_abandonment_rate"] > 0.7) + # High cart abandonment
        0.10 * (df["push_notification_open_rate"] < 0.2) +  # Low push engagement
        0.15 * (df["total_orders_last_6months"] == 0) +  # No recent purchases
        0.10 * (df["purchase_frequency_last_90days"] < 0.05) +  # Very low purchase frequency
        0.05 * np.random.random(n_samples)  # Random noise
    )
    
    # Convert to binary labels
    df["churn"] = (churn_probability > 0.5).astype(int)
    
    logger.info(f"Generated data with {df['churn'].sum()} churned users ({df['churn'].mean():.2%} churn rate)")
    
    return df

def load_data_from_aerospike(client):
    """Load training data from Aerospike (placeholder for real implementation)"""
    logger.info("Loading data from Aerospike...")
    
    # In a real implementation, this would:
    # 1. Scan all user features from Aerospike
    # 2. Combine with historical churn labels
    # 3. Create training dataset
    
    # For POC, we'll generate synthetic data
    return generate_synthetic_training_data()

def train_xgboost_model(df):
    """Train XGBoost model on the data"""
    logger.info("Training XGBoost model...")
    
    # Prepare features and target
    X = df[FEATURE_COLUMNS]
    y = df["churn"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Fit model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Model training completed. AUC Score: {auc_score:.3f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop Feature Importances:")
    logger.info(feature_importance.head())
    
    return model, auc_score, feature_importance

def save_model_and_metadata(model, auc_score, feature_importance):
    """Save trained model and metadata"""
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Save model
    model_file = os.path.join(MODEL_PATH, "churn_model.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}")
    
    # Save metadata
    metadata = {
        "model_type": "XGBoost",
        "training_timestamp": datetime.utcnow().isoformat(),
        "auc_score": float(auc_score),
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": feature_importance.to_dict('records')
    }
    
    metadata_file = os.path.join(MODEL_PATH, "model_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model metadata saved to {metadata_file}")

def main():
    """Main training pipeline"""
    logger.info("Starting model training pipeline...")
    
    # Connect to Aerospike
    client = connect_to_aerospike()
    
    try:
        # Load training data
        if client:
            df = load_data_from_aerospike(client)
        else:
            logger.warning("No Aerospike connection, using synthetic data only")
            df = generate_synthetic_training_data()
        
        # Train model
        model, auc_score, feature_importance = train_xgboost_model(df)
        
        # Save model and metadata
        save_model_and_metadata(model, auc_score, feature_importance)
        
        logger.info("Model training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise
    
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    main()
