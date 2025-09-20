"""
Feature Configuration for Churn Prediction Model
Shared between training and prediction services
"""

# Feature columns in the order expected by the model
FEATURE_COLUMNS = [
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

# Mapping from input feature names to model feature indices
FEATURE_MAPPING = {
    'acc_age_days': 0, 'member_dur': 1, 'days_last_login': 7, 'days_last_purch': 8,
    'sess_7d': 9, 'sess_30d': 10, 'avg_sess_dur': 11, 'ctr_10_sess': 12,
    'cart_abandon': 13, 'wishlist_ratio': 14, 'content_engage': 15,
    'avg_order_val': 16, 'orders_6m': 17, 'purch_freq_90d': 18, 'last_hv_purch': 19,
    'refund_rate': 20, 'discount_dep': 22, 'push_open_rate': 23, 'email_ctr': 24,
    'inapp_ctr': 25, 'promo_resp_time': 26, 'tickets_90d': 28, 'avg_ticket_res': 29,
    'csat_score': 30, 'refund_req': 31, 'curr_sess_clk': 32, 'checkout_time': 33,
    'cart_no_buy': 34, 'bounce_flag': 35
}

# Categorical feature mappings
CATEGORICAL_MAPPINGS = {
    'loyalty_tier': {'bronze': 1, 'silver': 2, 'gold': 3, 'platinum': 4},
    'geo_location': {'US-CA': 1, 'US-NY': 2, 'US-TX': 3, 'UK': 4, 'DE': 5},
    'device_type': {'mobile': 1, 'desktop': 2, 'tablet': 3},
    'pref_payment': {'credit': 1, 'debit': 2, 'paypal': 3, 'crypto': 4},
    'lang_pref': {'en': 1, 'es': 2, 'fr': 3, 'de': 4},
    'sub_pay_status': {'active': 1, 'inactive': 2, 'cancelled': 3},
    'retention_resp': {'positive': 1, 'negative': 2, 'neutral': 3}
}

# Categorical feature indices in the feature vector
CATEGORICAL_INDICES = {
    'loyalty_tier': 2, 'geo_location': 3, 'device_type': 4,
    'pref_payment': 5, 'lang_pref': 6, 'sub_pay_status': 21, 'retention_resp': 27
}

# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}

# Synthetic data generation parameters
SYNTHETIC_DATA_PARAMS = {
    'n_samples': 1000,
    'random_seed': 42
}
