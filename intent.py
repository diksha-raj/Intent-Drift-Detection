import os
from typing import List, Optional, Tuple
from transformers import pipeline

ZEROSHOT_MODEL = os.getenv("ZEROSHOT_MODEL", "facebook/bart-large-mnli")

DEFAULT_INTENTS = [
    "book_flight",
    "order_food",
    "check_weather",
    "customer_support",
    "schedule_meeting",
]

class IntentClassifier:
    def __init__(self):
        self._pipe = None

    def _ensure_pipe(self):
        if self._pipe is None:
            self._pipe = pipeline("zero-shot-classification", model=ZEROSHOT_MODEL, device=-1)

    def classify(self, text: str, candidate_intents: Optional[List[str]] = None) -> Tuple[str, float]:
        self._ensure_pipe()
        labels = candidate_intents or DEFAULT_INTENTS
        result = self._pipe(text, candidate_labels=labels, multi_label=False)
        # result has 'labels' in descending order of scores
        label = result["labels"][0]
        score = float(result["scores"][0])
        return label, score

    def fallback_keyword(self, text: str, candidate_intents: Optional[List[str]] = None) -> Tuple[Optional[str], float]:
        labels = candidate_intents or DEFAULT_INTENTS
        t = text.lower()
        mapping = {
            "book_flight": ["flight", "ticket", "airline", "book"],
            "order_food": ["pizza", "order", "food", "burger"],
            "check_weather": ["weather", "rain", "temperature", "forecast"],
            "customer_support": ["support", "help", "issue", "problem"],
            "schedule_meeting": ["meeting", "schedule", "calendar", "invite"],
        }
        for label in labels:
            for kw in mapping.get(label, []):
                if kw in t:
                    return label, 0.6
        return None, 0.0
