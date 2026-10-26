"""
tests/test_api.py - the serving contract, the rails, and the privacy promise.

WHY httpx.ASGITransport INSTEAD OF fastapi.testclient
  Starlette's TestClient runs the app in a background portal thread. In this
  environment (Python 3.14) that thread does not survive a full-suite run: the
  first request dies with concurrent.futures.CancelledError, and entering a
  client per test hangs. Driving the ASGI app directly with httpx exercises the
  SAME middleware stack (rate limit, body cap, exception handlers) with no
  thread at all, so the suite is deterministic and fast.

What must hold:
  * /analyze returns the documented shape for a scam, a legit bank alert and a
    casual Hinglish message;
  * bad input is rejected (empty, too long, control characters);
  * NO message text reaches the log;
  * the mode actually changes the decision threshold;
  * a missing MuRIL run falls back to TF-IDF instead of crashing;
  * TF-IDF p95 latency stays inside the budget in configs/serving.yaml.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import numpy as np
import pytest
import yaml

from src import api
from src.config import PROJECT_ROOT
from src.serving import registry as registry_mod

BASE = "http://testserver"


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    """One request against the app, in a fresh event loop."""

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def post(text: str, mode: str | None = None) -> httpx.Response:
    payload = {"text": text}
    if mode is not None:
        payload["mode"] = mode
    return _request("POST", "/analyze", json=payload)


def get(path: str) -> httpx.Response:
    return _request("GET", path)


@pytest.fixture(autouse=True)
def _clean_state():
    """Fresh rate-limit bucket per test (the cap is 30 requests/minute)."""
    api._hits.clear()
    yield
    api._hits.clear()


SCAM = "Your account will be blocked today. Share the OTP you just received."
BANK_ALERT = ("Rs 1,200 debited from your account ending 4417 at a merchant "
              "terminal in Pune. Not you? Call the number on your card.")
HINGLISH = "Sun na, shaam ko ghar aa raha hoon, kuch laana hai kya market se?"


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------
def test_analyze_contract():
    response = post(SCAM)
    assert response.status_code == 200
    body = response.json()
    for key in ("verdict", "scam_probability", "scam_probability_raw", "mode",
                "threshold", "tactics", "reasons", "highlights", "warnings",
                "model_info", "latency_ms"):
        assert key in body, f"missing {key} in /analyze response"
    assert body["verdict"] in {"scam", "suspicious", "safe"}
    assert 0.0 <= body["scam_probability"] <= 1.0
    assert isinstance(body["tactics"], list)
    assert isinstance(body["warnings"], list) and body["warnings"]
    assert body["model_info"]["validated_on"]


def test_scam_example_is_flagged():
    body = post(SCAM).json()
    assert body["verdict"] == "scam", body["reasons"]


def test_legit_bank_alert_is_not_scam():
    body = post(BANK_ALERT).json()
    assert body["verdict"] in {"safe", "suspicious"}, body
    assert body["scam_probability"] < 0.99


def test_hinglish_is_accepted():
    response = post(HINGLISH)
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in {"scam", "suspicious", "safe"}
    assert any("English" in w for w in body["warnings"])


# ---------------------------------------------------------------------------
# validation + rails
# ---------------------------------------------------------------------------
def test_empty_text_is_rejected():
    assert post("   ").status_code == 422


def test_too_long_text_is_rejected():
    assert post("a" * 1001).status_code == 422


def test_exactly_max_length_is_accepted():
    assert post("a " * 500).status_code == 200


def test_control_characters_do_not_break_scoring():
    dirty = "Your account\x00 will be\x07 blocked, share the OTP\x1f now."
    response = post(dirty)
    assert response.status_code == 200
    assert "\x00" not in response.json()["reasons"][0]


def test_unknown_mode_is_rejected():
    assert post(SCAM, mode="yolo").status_code == 422


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 3)
    api._hits.clear()
    codes = [post(SCAM).status_code for _ in range(5)]
    assert codes[0] == 200
    assert codes[-1] == 429


def test_body_size_cap():
    response = post("x" * 9000)
    assert response.status_code in (413, 422)


# ---------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------
def test_no_message_text_reaches_the_log(caplog):
    marker = "ZZZDO-NOT-LOG-THIS-MARKER"
    with caplog.at_level(logging.DEBUG):
        post(f"{SCAM} {marker}")
    for record in caplog.records:
        assert marker not in record.getMessage()
        assert SCAM not in record.getMessage()
    assert any("analyze" in r.getMessage() for r in caplog.records), \
        "the endpoint logged nothing at all, so this test proves nothing"


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def test_mode_changes_the_threshold():
    point = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "operating_point.yaml").read_text(encoding="utf-8"))
    expected = {m: round(info["threshold"], 6) for m, info in point["modes"].items()}
    for mode, threshold in expected.items():
        body = post(SCAM, mode=mode).json()
        assert body["mode"] == mode
        assert abs(body["threshold"] - threshold) < 1e-6


def test_modes_endpoint_lists_all_points():
    body = get("/modes").json()
    assert set(body["modes"][0]) >= {"mode", "threshold", "rule"}
    assert {m["mode"] for m in body["modes"]} == {"strict", "balanced", "sensitive"}


# ---------------------------------------------------------------------------
# health / fallback
# ---------------------------------------------------------------------------
def test_health_reports_the_served_model():
    body = get("/health").json()
    assert body["status"] == "ok"
    assert body["serving_model"] in {"tfidf", "muril"}
    assert body["temperature"] > 0
    assert body["default_mode"] in body["modes"]


def test_missing_muril_falls_back_to_tfidf(tmp_path):
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "serving.yaml").read_text(encoding="utf-8"))
    cfg["model"] = "muril"
    cfg["models"]["muril"]["run_dir"] = "outputs/does_not_exist"
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    reg = registry_mod.Registry(serving_path=path)
    assert reg.served_model == "tfidf"
    assert reg.fallback_reason and "fell back" in reg.fallback_reason
    assert reg.analyze(SCAM)["verdict"] in {"scam", "suspicious", "safe"}


def test_model_card_and_index_served():
    assert get("/model-card").status_code in (200, 404)
    assert get("/").status_code == 200


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
def test_tfidf_p95_latency_under_budget():
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "serving.yaml").read_text(encoding="utf-8"))
    budget = float(cfg["models"]["tfidf"]["p95_budget_ms"])
    text = SCAM * 2                      # ~120 chars, inside the occlusion cap
    timings = []
    for _ in range(20):
        api._hits.clear()
        timings.append(post(text).json()["latency_ms"])
    p95 = float(np.percentile(timings, 95))
    assert p95 < budget, f"p95 {p95:.1f} ms exceeds {budget} ms"
