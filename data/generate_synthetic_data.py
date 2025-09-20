import json
import random
import numpy as np
from datetime import datetime, timedelta
import requests
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
NUM_USERS = 100

# Sample data pools
LOYALTY_TIERS = ["bronze", "silver", "gold", "platinum"]
GEO_LOCATIONS = ["US-CA", "US-NY", "US-TX", "UK-LON", "DE-BER", "FR-PAR"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
PAYMENT_METHODS = ["credit_card", "paypal", "apple_pay", "google_pay"]
LANGUAGES = ["en", "es", "fr", "de", "it"]
SUBSCRIPTION_STATUS = ["active", "expired", "cancelled", "pending"]

def generate_user_profile_features(user_id):
    """Generate realistic user profile features"""
    return {
        "user_id": user_id,
        "account_age_days": random.randint(30, 1000),
        "membership_duration": random.randint(1, 36),
        "loyalty_tier": random.choice(LOYALTY_TIERS),
        "geo_location": random.choice(GEO_LOCATIONS),
        "device_type": random.choice(DEVICE_TYPES),
        "preferred_payment_method": random.choice(PAYMENT_METHODS),
        "language_preference": random.choice(LANGUAGES)
    }

def generate_user_behavior_features(user_id):
    """Generate realistic user behavior features"""
    # Create correlated behavior patterns
    is_active_user = random.random() > 0.3  # 70% active users
    
    if is_active_user:
        days_since_login = random.randint(0, 3)
        days_since_purchase = random.randint(0, 14)
        sessions_7days = random.randint(5, 15)
        sessions_30days = random.randint(15, 50)
    else:
        days_since_login = random.randint(7, 30)
        days_since_purchase = random.randint(30, 90)
        sessions_7days = random.randint(0, 3)
        sessions_30days = random.randint(0, 10)
    
    return {
        "user_id": user_id,
        "days_since_last_login": days_since_login,
        "days_since_last_purchase": days_since_purchase,
        "sessions_last_7days": sessions_7days,
        "sessions_last_30days": sessions_30days,
        "avg_session_duration_last_30days": random.uniform(2.0, 45.0),
        "click_through_rate_last_10_sessions": random.uniform(0.05, 0.25),
        "cart_abandonment_rate": random.uniform(0.1, 0.8),
        "wishlist_adds_vs_purchases": random.uniform(0.1, 2.0),
        "content_engagement_rate": random.uniform(0.1, 0.6)
    }

def generate_transactional_features(user_id):
    """Generate realistic transactional features"""
    is_high_value = random.random() > 0.8  # 20% high-value customers
    
    if is_high_value:
        avg_order_value = random.uniform(100, 500)
        total_orders = random.randint(10, 50)
        purchase_frequency = random.uniform(0.1, 0.3)
    else:
        avg_order_value = random.uniform(20, 100)
        total_orders = random.randint(0, 15)
        purchase_frequency = random.uniform(0.01, 0.1)
    
    return {
        "user_id": user_id,
        "avg_order_value": round(avg_order_value, 2),
        "total_orders_last_6months": total_orders,
        "purchase_frequency_last_90days": round(purchase_frequency, 3),
        "time_since_last_high_value_purchase": random.randint(0, 180),
        "refund_rate": random.uniform(0.0, 0.15),
        "subscription_payment_status": random.choice(SUBSCRIPTION_STATUS),
        "discount_dependency_score": random.uniform(0.1, 0.9),
        "category_spend_distribution": {
            "electronics": random.uniform(0.1, 0.4),
            "clothing": random.uniform(0.1, 0.3),
            "books": random.uniform(0.05, 0.2),
            "home": random.uniform(0.1, 0.25)
        }
    }

def generate_engagement_features(user_id):
    """Generate realistic engagement features"""
    engagement_level = random.choice(["low", "medium", "high"])
    
    if engagement_level == "high":
        push_open_rate = random.uniform(0.4, 0.8)
        email_click_rate = random.uniform(0.3, 0.6)
        offer_click_rate = random.uniform(0.2, 0.5)
    elif engagement_level == "medium":
        push_open_rate = random.uniform(0.2, 0.4)
        email_click_rate = random.uniform(0.1, 0.3)
        offer_click_rate = random.uniform(0.1, 0.2)
    else:
        push_open_rate = random.uniform(0.0, 0.2)
        email_click_rate = random.uniform(0.0, 0.1)
        offer_click_rate = random.uniform(0.0, 0.1)
    
    return {
        "user_id": user_id,
        "push_notification_open_rate": round(push_open_rate, 3),
        "email_click_rate": round(email_click_rate, 3),
        "in_app_offer_click_rate": round(offer_click_rate, 3),
        "response_time_to_promotions": random.uniform(0.5, 24.0),
        "recent_retention_offer_response": random.choice(["accepted", "declined", "ignored", None])
    }

def generate_support_features(user_id):
    """Generate realistic support features"""
    has_issues = random.random() > 0.7  # 30% of users have support issues
    
    if has_issues:
        tickets = random.randint(1, 8)
        resolution_time = random.uniform(2.0, 48.0)
        csat_score = random.uniform(2.0, 4.0)
        refund_requests = random.randint(0, 3)
    else:
        tickets = 0
        resolution_time = random.uniform(0.5, 4.0)
        csat_score = random.uniform(4.0, 5.0)
        refund_requests = 0
    
    return {
        "user_id": user_id,
        "support_tickets_last_90days": tickets,
        "avg_ticket_resolution_time": round(resolution_time, 1),
        "csat_score_last_interaction": round(csat_score, 1),
        "refund_requests": refund_requests
    }

def generate_realtime_features(user_id):
    """Generate realistic real-time session features"""
    is_engaged_session = random.random() > 0.4  # 60% engaged sessions
    
    return {
        "user_id": user_id,
        "current_session_clicks": random.randint(1, 50) if is_engaged_session else random.randint(0, 5),
        "time_spent_on_checkout_page": random.uniform(0.0, 300.0),
        "added_to_cart_but_not_bought_flag": random.choice([True, False]),
        "session_bounce_flag": not is_engaged_session
    }

def ingest_features_to_api(features, endpoint):
    """Send features to the API"""
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", json=features)
        if response.status_code == 200:
            return True
        else:
            print(f"Error ingesting {endpoint} for user {features['user_id']}: {response.text}")
            return False
    except Exception as e:
        print(f"Failed to ingest {endpoint} for user {features['user_id']}: {str(e)}")
        return False

def generate_and_ingest_user_data(user_id):
    """Generate and ingest all feature types for a user"""
    print(f"Generating data for user {user_id}...")
    
    # Generate all feature types
    profile_features = generate_user_profile_features(user_id)
    behavior_features = generate_user_behavior_features(user_id)
    transactional_features = generate_transactional_features(user_id)
    engagement_features = generate_engagement_features(user_id)
    support_features = generate_support_features(user_id)
    realtime_features = generate_realtime_features(user_id)
    
    # Ingest to API
    success_count = 0
    
    if ingest_features_to_api(profile_features, "ingest/profile"):
        success_count += 1
    
    if ingest_features_to_api(behavior_features, "ingest/behavior"):
        success_count += 1
    
    if ingest_features_to_api(transactional_features, "ingest/transactional"):
        success_count += 1
    
    if ingest_features_to_api(engagement_features, "ingest/engagement"):
        success_count += 1
    
    if ingest_features_to_api(support_features, "ingest/support"):
        success_count += 1
    
    if ingest_features_to_api(realtime_features, "ingest/realtime"):
        success_count += 1
    
    return success_count == 6

def test_prediction_api(user_id):
    """Test the prediction API for a user"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict/{user_id}")
        if response.status_code == 200:
            prediction = response.json()
            print(f"Prediction for user {user_id}:")
            print(f"  Churn Probability: {prediction['churn_probability']}")
            print(f"  Risk Segment: {prediction['risk_segment']}")
            print(f"  Churn Reasons: {prediction['churn_reasons']}")
            return True
        else:
            print(f"Prediction failed for user {user_id}: {response.text}")
            return False
    except Exception as e:
        print(f"Failed to get prediction for user {user_id}: {str(e)}")
        return False

def main():
    """Main function to generate synthetic data"""
    print(f"Generating synthetic data for {NUM_USERS} users...")
    print(f"API Base URL: {API_BASE_URL}")
    
    successful_users = 0
    
    for i in range(1, NUM_USERS + 1):
        user_id = f"user_{i:04d}"
        
        if generate_and_ingest_user_data(user_id):
            successful_users += 1
            
            # Test prediction for every 10th user
            if i % 10 == 0:
                print(f"\nTesting prediction for {user_id}...")
                test_prediction_api(user_id)
                print()
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    print(f"\nData generation completed!")
    print(f"Successfully processed {successful_users}/{NUM_USERS} users")
    
    # Test a few predictions
    print("\nTesting predictions for sample users...")
    for user_id in ["user_0001", "user_0025", "user_0050", "user_0075", "user_0100"]:
        test_prediction_api(user_id)
        print()

if __name__ == "__main__":
    main()
