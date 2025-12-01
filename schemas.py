from pydantic import BaseModel, Field
from typing import List, Optional

class SessionStartRequest(BaseModel):
    initial_message: str = Field(..., min_length=1)
    candidate_intents: Optional[List[str]] = None

class SessionStartResponse(BaseModel):
    session_id: str
    initial_intent: Optional[str] = None
    confidence: Optional[float] = None

class DetectRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)
    candidate_intents: Optional[List[str]] = None

class DetectResponse(BaseModel):
    current_intent: Optional[str]
    intent_confidence: Optional[float]
    drift_score: float
    smoothed_drift: float
    similarity: float
    drift_alert: bool
