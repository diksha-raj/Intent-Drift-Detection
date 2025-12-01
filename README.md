# Intent Drift Detection (FastAPI + Open-Source LLM)

An end-to-end, local, no-cost hobby project to detect changes in user intent during a conversation. Built with FastAPI, sentence-transformers for embeddings, and zero-shot intent classification. Runs fully offline (models cached locally) and uses only open-source models.

## Features

- FastAPI service with simple endpoints
- Intent drift detection using text embeddings and cosine similarity
- Zero-shot intent classification to label intents
- Session tracking: initial intent, current intent, drift score, and alerts
- Configurable thresholds and smoothing
- Unit tests and a tiny smoke test


## Requirements

- macOS (or Linux/Windows) with Python 3.9+
- No paid services; models are open-source from HuggingFace

## Quick Start

1. Create and activate a virtual environment
2. Install dependencies
3. Run the API server
4. Call the endpoints (examples below)

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --reload --port 8000
```

### API

- `GET /ping` — health check
- `POST /session` — start/reset session with an initial message
- `POST /detect` — detect drift for a new message in the current session

#### Start/Reset Session

Request:
```json
{
  "initial_message": "I need help booking a flight",
  "candidate_intents": ["book_flight", "order_food", "check_weather"]
}
```

Response:
```json
{
  "session_id": "abc123",
  "initial_intent": "book_flight",
  "confidence": 0.81
}
```

#### Detect Drift

Request:
```json
{
  "session_id": "abc123",
  "message": "Actually I want pizza",
  "candidate_intents": ["book_flight", "order_food", "check_weather"]
}
```

Response:
```json
{
  "current_intent": "order_food",
  "intent_confidence": 0.77,
  "drift_score": 0.62,
  "drift_alert": true
}
```

## How drift is measured

- Embed the initial message and the latest message using `sentence-transformers/all-MiniLM-L6-v2`
- Compute cosine similarity; drift score = 1 - similarity
- Apply exponential smoothing to stabilize; compare against threshold (default 0.45)
- If drift score exceeds threshold and current intent label differs from initial intent, raise alert

## Configuration

Use environment variables in `.env` or system env:

- `EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `ZEROSHOT_MODEL=facebook/bart-large-mnli`
- `DRIFT_THRESHOLD=0.45`
- `SMOOTHING_ALPHA=0.4`

## Local-only operation

All models are downloaded once and cached under `~/.cache/huggingface/`. No paid services used.

> Optional: If you prefer fully offline (no first-time download), manually place model folders and set `HF_HOME` to that directory.

## Testing

```bash
pytest -q
```
