# ScamLens serving image - CPU only, non-root.
#
# The image does NOT ship data or model weights (they are gitignored and must
# never be baked into a public image). Build the artefacts on the host first:
#
#   python -m src.build_dataset && python -m src.baselines
#   python -m src.calibrate --model verdict_lr_plus
#   docker build -t scamlens:0.5 .
#
# `models/` and `configs/` are copied in; `data/` is not, because serving only
# needs the fitted thresholds, not the messages.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal: curl is for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so pip does not pull a 2 GB CUDA wheel.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt \
    && pip install "fastapi>=0.110" "uvicorn[standard]>=0.29"

COPY src/ ./src/
COPY configs/ ./configs/
COPY static/ ./static/
COPY docs/ ./docs/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Non-root: the service needs no write access to its own image.
RUN useradd --create-home --uid 10001 scamlens \
    && chown -R scamlens:scamlens /app
USER scamlens

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
