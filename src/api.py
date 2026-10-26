"""
src/api.py - the ScamLens HTTP service (Phase 5).

ENDPOINTS
  POST /analyze      {text, mode?} -> verdict, probability, tactics, reasons,
                     highlights, warnings
  GET  /health       which model is actually serving, thresholds, limits
  GET  /model-card   docs/MODEL_CARD.md
  GET  /modes        the three operating points with measured recall / FPR
  GET  /             the demo UI (static/index.html)

WHAT IS DELIBERATELY MISSING
  No message text is logged, cached, stored or returned in telemetry. The only
  thing written to the log is length, verdict, mode and latency - the test suite
  asserts that a marker string in the request never reaches the log.

RAILS
  * input: 1-1000 characters after control-character stripping;
  * body: 8 KiB cap (checked before parsing);
  * rate: N requests/minute per client IP (configs/serving.yaml).

Run:
  uvicorn src.api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT
from src.serving.registry import get_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("scamlens.api")

MAX_BODY_BYTES = 8192
RATE_LIMIT_PER_MIN = 30
_WINDOW_SECONDS = 60.0
_hits: dict[str, deque] = defaultdict(deque)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Message to analyse (1-1000 characters)")
    mode: str | None = Field(
        None, description="strict | balanced | sensitive (default from config)")


class Tactic(BaseModel):
    tag: str
    prob: float
    validated: bool


class Highlight(BaseModel):
    word: str
    weight: float


class ModelInfo(BaseModel):
    name: str
    key: str
    mode: str
    training_mix: str | None = None
    validated_on: str | None = None
    temperature: float


class AnalyzeResponse(BaseModel):
    verdict: str
    scam_probability: float
    scam_probability_raw: float
    mode: str
    threshold: float
    tactics: list[Tactic]
    reasons: list[str]
    highlights: list[Highlight]
    warnings: list[str]
    model_info: ModelInfo
    latency_ms: float


# ---------------------------------------------------------------------------
# app + rails
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ScamLens",
    version="0.5.0",
    description="Research prototype for Indian SMS/WhatsApp scam triage. "
                "Not a substitute for verifying with your bank.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # research prototype; no cookies or auth
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def rails(request: Request, call_next):
    """Body-size cap, per-IP rate limit, and a text-free access log."""
    started = time.perf_counter()
    length = request.headers.get("content-length")
    if length and int(length) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "request body too large"}, status_code=413)

    if request.url.path == "/analyze":
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = _hits[ip]
        while bucket and now - bucket[0] > _WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MIN:
            return JSONResponse(
                {"detail": "rate limit exceeded, try again in a minute"},
                status_code=429)
        bucket.append(now)

    response = await call_next(request)
    # Access log: path, status, latency. NEVER the body.
    log.info("path=%s status=%s latency_ms=%.1f",
             request.url.path, response.status_code,
             (time.perf_counter() - started) * 1000)
    return response


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, request: Request):
    """Score one message. The text is used and discarded."""
    registry = get_registry()
    try:
        result = registry.analyze(payload.text, payload.mode)
    except ValueError as exc:                     # validation from the registry
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Telemetry without content: length, verdict, mode, latency only.
    log.info("analyze len=%d verdict=%s mode=%s latency_ms=%.1f",
             len(payload.text), result["verdict"], result["mode"],
             result["latency_ms"])
    return result


@app.get("/health")
def health():
    return get_registry().health()


@app.get("/modes")
def modes():
    return {"default": get_registry().default_mode,
            "modes": get_registry().mode_table()}


@app.get("/model-card", response_class=PlainTextResponse)
def model_card():
    path = PROJECT_ROOT / "docs" / "MODEL_CARD.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="MODEL_CARD.md not found")
    return path.read_text(encoding="utf-8")


STATIC_DIR = Path(PROJECT_ROOT) / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    path = STATIC_DIR / "index.html"
    if not path.exists():
        return JSONResponse({"detail": "UI not built", "health": "/health"})
    return FileResponse(path)
