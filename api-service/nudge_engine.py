from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models
class NudgeRequest(BaseModel):
    user_id: str
    churn_probability: float
    risk_segment: str
    churn_reasons: List[str]

class NudgeAction(BaseModel):
    type: str
    content_template: str
    channel: str
    priority: int

class NudgeResponse(BaseModel):
    user_id: str
    nudges_triggered: List[NudgeAction]
    rule_matched: str
    timestamp: str

# Nudge Rules as defined in the plan
NUDGE_RULES = [
    {
        "rule_id": "rule_1",
        "churn_score_range": [0.6, 0.8],
        "churn_reasons": ["INACTIVITY", "DELIVERY_ISSUES"],
        "nudges": [
            {"type": "Email", "content_template": "Template 1", "channel": "email", "priority": 1}
        ]
    },
    {
        "rule_id": "rule_2", 
        "churn_score_range": [0.8, 1.0],
        "churn_reasons": ["CART_ABANDONMENT"],
        "nudges": [
            {"type": "Push Notification", "content_template": "Template 2", "channel": "push", "priority": 1},
            {"type": "Discount Coupon", "content_template": "Template 2", "channel": "email", "priority": 2}
        ]
    },
    {
        "rule_id": "rule_3",
        "churn_score_range": [0.7, 0.9],
        "churn_reasons": ["LOW_ENGAGEMENT"],
        "nudges": [
            {"type": "Email", "content_template": "Template 3", "channel": "email", "priority": 1}
        ]
    },
    {
        "rule_id": "rule_4",
        "churn_score_range": [0.6, 0.75],
        "churn_reasons": ["PRICE_SENSITIVITY"],
        "nudges": [
            {"type": "Discount Coupon", "content_template": "Template 4", "channel": "email", "priority": 1}
        ]
    },
    {
        "rule_id": "rule_5",
        "churn_score_range": [0.85, 1.0],
        "churn_reasons": ["PAYMENT_FAILURE"],
        "nudges": [
            {"type": "Push Notification", "content_template": "Template 5", "channel": "push", "priority": 1},
            {"type": "Email", "content_template": "Template 5", "channel": "email", "priority": 2}
        ]
    },
    {
        "rule_id": "rule_6",
        "churn_score_range": [0.65, 0.8],
        "churn_reasons": ["PRODUCT_AVAILABILITY"],
        "nudges": [
            {"type": "Push Notification", "content_template": "Template 6", "channel": "push", "priority": 1}
        ]
    },
    {
        "rule_id": "rule_7",
        "churn_score_range": [0.7, 0.9],
        "churn_reasons": ["INACTIVITY"],
        "nudges": [
            {"type": "Push Notification", "content_template": "Template 7", "channel": "push", "priority": 1}
        ]
    },
    {
        "rule_id": "rule_8",
        "churn_score_range": [0.6, 0.8],
        "churn_reasons": ["CART_ABANDONMENT", "LOW_ENGAGEMENT"],
        "nudges": [
            {"type": "Email", "content_template": "Template 8", "channel": "email", "priority": 1},
            {"type": "Discount Coupon", "content_template": "Template 8", "channel": "email", "priority": 2}
        ]
    },
    {
        "rule_id": "rule_9",
        "churn_score_range": [0.75, 0.95],
        "churn_reasons": ["DELIVERY_ISSUES", "PRICE_SENSITIVITY"],
        "nudges": [
            {"type": "Push Notification", "content_template": "Template 9", "channel": "push", "priority": 1}
        ]
    },
    {
        "rule_id": "rule_10",
        "churn_score_range": [0.8, 1.0],
        "churn_reasons": ["PAYMENT_FAILURE", "CART_ABANDONMENT"],
        "nudges": [
            {"type": "Push Notification", "content_template": "Template 10", "channel": "push", "priority": 1},
            {"type": "Discount Coupon", "content_template": "Template 10", "channel": "email", "priority": 2},
            {"type": "Email", "content_template": "Template 10", "channel": "email", "priority": 3}
        ]
    }
]

class NudgeEngine:
    """Nudge engine for triggering retention nudges based on churn predictions"""
    
    def __init__(self):
        self.rules = NUDGE_RULES
        logger.info(f"Nudge engine initialized with {len(self.rules)} rules")
    
    def find_matching_rule(self, churn_probability: float, churn_reasons: List[str]) -> Dict[str, Any]:
        """Find the first matching nudge rule based on churn score and reasons"""
        
        # Sort rules by priority (rule_10 has highest priority)
        sorted_rules = sorted(self.rules, key=lambda x: int(x["rule_id"].split("_")[1]), reverse=True)
        
        for rule in sorted_rules:
            # Check if churn probability is in range
            score_min, score_max = rule["churn_score_range"]
            if not (score_min <= churn_probability <= score_max):
                continue
            
            # Check if any churn reason matches
            rule_reasons = rule["churn_reasons"]
            if any(reason in rule_reasons for reason in churn_reasons):
                return rule
        
        return None
    
    def execute_nudges(self, user_id: str, nudges: List[Dict[str, Any]]) -> List[NudgeAction]:
        """Execute nudges (for POC, just log them)"""
        executed_nudges = []
        
        for nudge in nudges:
            # In production, this would actually send emails, push notifications, etc.
            # For POC, we just log the action
            logger.info(f"NUDGE EXECUTED - User: {user_id}, Type: {nudge['type']}, "
                       f"Channel: {nudge['channel']}, Template: {nudge['content_template']}")
            
            executed_nudges.append(NudgeAction(
                type=nudge["type"],
                content_template=nudge["content_template"],
                channel=nudge["channel"],
                priority=nudge["priority"]
            ))
        
        return executed_nudges
    
    def trigger_nudges(self, user_id: str, churn_probability: float, risk_segment: str, churn_reasons: List[str]) -> NudgeResponse:
        """Trigger nudges based on churn score and reasons"""
        logger.info(f"Processing nudge request for user {user_id} "
                   f"(score: {churn_probability}, segment: {risk_segment})")
        
        # Find matching rule
        matching_rule = self.find_matching_rule(churn_probability, churn_reasons)
        
        if not matching_rule:
            logger.info(f"No matching nudge rule found for user {user_id}")
            return NudgeResponse(
                user_id=user_id,
                nudges_triggered=[],
                rule_matched="none",
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Execute nudges
        executed_nudges = self.execute_nudges(user_id, matching_rule["nudges"])
        
        logger.info(f"Triggered {len(executed_nudges)} nudges for user {user_id} "
                   f"using {matching_rule['rule_id']}")
        
        return NudgeResponse(
            user_id=user_id,
            nudges_triggered=executed_nudges,
            rule_matched=matching_rule["rule_id"],
            timestamp=datetime.utcnow().isoformat()
        )
    
    def get_rules(self) -> Dict[str, Any]:
        """Get all nudge rules"""
        return {"rules": self.rules, "total_rules": len(self.rules)}
    
    def get_rule(self, rule_id: str) -> Dict[str, Any]:
        """Get specific nudge rule by ID"""
        for rule in self.rules:
            if rule["rule_id"] == rule_id:
                return rule
        return None
    
    def test_rules(self, user_id: str, churn_probability: float, churn_reasons: List[str]) -> Dict[str, Any]:
        """Test which rule would match for given parameters"""
        matching_rule = self.find_matching_rule(churn_probability, churn_reasons)
        
        if not matching_rule:
            return {
                "user_id": user_id,
                "matching_rule": None,
                "message": "No rule matches the given parameters"
            }
        
        return {
            "user_id": user_id,
            "matching_rule": matching_rule["rule_id"],
            "rule_details": matching_rule,
            "would_trigger": len(matching_rule["nudges"])
        }

# Global nudge engine instance
nudge_engine = NudgeEngine()

def get_nudge_health() -> Dict[str, Any]:
    """Get nudge engine health status"""
    return {
        "rules_loaded": len(nudge_engine.rules),
        "timestamp": datetime.utcnow().isoformat()
    }
