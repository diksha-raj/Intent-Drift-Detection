import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import SessionStartRequest, SessionStartResponse, DetectRequest, DetectResponse
from .drift import DriftDetector
from .intent import IntentClassifier

app = FastAPI(title="Intent Drift Detection", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

drift = DriftDetector()
intent = IntentClassifier()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/session", response_model=SessionStartResponse)
def start_session(payload: SessionStartRequest):
    session_id = str(uuid.uuid4())
    drift.start_session(session_id, payload.initial_message)
    # Try LLM intent classification, fallback to keywords if pipeline fails
    try:
        label, score = intent.classify(payload.initial_message, payload.candidate_intents)
    except Exception:
        label, score = intent.fallback_keyword(payload.initial_message, payload.candidate_intents)
    return SessionStartResponse(session_id=session_id, initial_intent=label, confidence=score)

@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    try:
        sim, drift_score, smoothed, alert = drift.compute(payload.session_id, payload.message)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # classify current message
    try:
        curr_label, curr_score = intent.classify(payload.message, payload.candidate_intents)
    except Exception:
        curr_label, curr_score = intent.fallback_keyword(payload.message, payload.candidate_intents)
    return DetectResponse(
        current_intent=curr_label,
        intent_confidence=curr_score,
        drift_score=drift_score,
        smoothed_drift=smoothed,
        similarity=sim,
        drift_alert=alert,
    )
