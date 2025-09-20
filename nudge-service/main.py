from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nudge Service", version="1.0.0")

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

def find_matching_rule(churn_probability: float, churn_reasons: List[str]) -> Dict[str, Any]:
    """Find the first matching nudge rule based on churn score and reasons"""
    
    # Sort rules by priority (rule_10 has highest priority)
    sorted_rules = sorted(NUDGE_RULES, key=lambda x: int(x["rule_id"].split("_")[1]), reverse=True)
    
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

def execute_nudges(user_id: str, nudges: List[Dict[str, Any]]) -> List[NudgeAction]:
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

@app.get("/")
async def root():
    return {"message": "Nudge Service", "version": "1.0.0"}

@app.post("/trigger")
async def trigger_nudge(request: NudgeRequest) -> NudgeResponse:
    """Trigger nudges based on churn score and reasons"""
    try:
        logger.info(f"Processing nudge request for user {request.user_id} "
                   f"(score: {request.churn_probability}, segment: {request.risk_segment})")
        
        # Find matching rule
        matching_rule = find_matching_rule(request.churn_probability, request.churn_reasons)
        
        if not matching_rule:
            logger.info(f"No matching nudge rule found for user {request.user_id}")
            return NudgeResponse(
                user_id=request.user_id,
                nudges_triggered=[],
                rule_matched="none",
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Execute nudges
        executed_nudges = execute_nudges(request.user_id, matching_rule["nudges"])
        
        logger.info(f"Triggered {len(executed_nudges)} nudges for user {request.user_id} "
                   f"using {matching_rule['rule_id']}")
        
        return NudgeResponse(
            user_id=request.user_id,
            nudges_triggered=executed_nudges,
            rule_matched=matching_rule["rule_id"],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error triggering nudges for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Nudge trigger failed: {str(e)}")

@app.get("/rules")
async def get_nudge_rules():
    """Get all nudge rules"""
    return {"rules": NUDGE_RULES, "total_rules": len(NUDGE_RULES)}

@app.get("/rules/{rule_id}")
async def get_nudge_rule(rule_id: str):
    """Get specific nudge rule by ID"""
    for rule in NUDGE_RULES:
        if rule["rule_id"] == rule_id:
            return rule
    
    raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")

@app.post("/test/{user_id}")
async def test_nudge_rules(user_id: str, churn_probability: float, churn_reasons: List[str]):
    """Test which rule would match for given parameters"""
    matching_rule = find_matching_rule(churn_probability, churn_reasons)
    
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rules_loaded": len(NUDGE_RULES),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Default values
    host = "0.0.0.0"
    port = 8002
    
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
