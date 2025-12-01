import os
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.4"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.45"))

class DriftDetector:
    def __init__(self):
        self.model = None
        self.sessions = {}

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def start_session(self, session_id: str, initial_message: str):
        self._ensure_model()
        init_vec = self.model.encode([initial_message])
        self.sessions[session_id] = {
            "initial_message": initial_message,
            "initial_vec": init_vec,
            "last_smoothed": 0.0,
        }

    def compute(self, session_id: str, message: str) -> Tuple[float, float, float, bool]:
        self._ensure_model()
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        init_vec = session["initial_vec"]
        msg_vec = self.model.encode([message])
        sim = float(cosine_similarity(init_vec, msg_vec)[0][0])
        drift = 1.0 - sim
        smoothed = SMOOTHING_ALPHA * drift + (1 - SMOOTHING_ALPHA) * session["last_smoothed"]
        session["last_smoothed"] = smoothed
        alert = smoothed >= DRIFT_THRESHOLD
        return sim, drift, smoothed, alert
