"""
src/serving/registry.py - load the model ONCE, decide per request.

WHAT THIS OWNS
  * which model is served (configs/serving.yaml: tfidf | muril) and the
    fallback when the chosen one is missing;
  * the temperature and the three operating points (configs/operating_point.yaml,
    fitted on validation only - this module never fits anything);
  * ONE analyze() call that turns text into the response the API returns:
    verdict, calibrated probability, tactics, reasons, highlights, warnings.

PRIVACY
  analyze() takes text and returns derived numbers. It does not log, cache or
  write the text anywhere, and the API layer only logs length / verdict /
  latency / mode. tests/test_api.py proves that.

WHY THE VERDICT IS NOT JUST "p > threshold"
  The scorer is bimodal, so Phase 5 added a deferral rule (src/suspicious.py):
  a high score with no nameable evidence becomes `suspicious`, not `scam`.
  Serving and evaluation share that rule, so the API cannot drift from the
  numbers in results/.

Run:
  python -m src.serving.registry            # loads everything and self-checks
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import yaml

from src.config import CFG, PROJECT_ROOT
from src.phase4_common import apply_temperature
from src.suspicious import EVIDENCE_PATTERNS, RISKY_URL_TYPES, decide, rule_evidence

SERVING_CFG_PATH = PROJECT_ROOT / "configs" / "serving.yaml"
POINT_PATH = PROJECT_ROOT / "configs" / "operating_point.yaml"

# Control characters (except tab/newline) are stripped before scoring: they are
# never part of a real message and they make tokenisation unpredictable.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F'’-]{2,}")
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|in|info|"
                    r"xyz|top|co|tk|ly|site|online)\b)", re.IGNORECASE)


def sanitize_text(text: str) -> str:
    """Strip control characters and collapse whitespace. No other rewriting."""
    text = unicodedata.normalize("NFC", str(text))
    text = CONTROL_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


class Registry:
    """Loaded models + operating points. Build once, reuse for every request."""

    def __init__(self, serving_path: Path | str = SERVING_CFG_PATH,
                 point_path: Path | str = POINT_PATH) -> None:
        self.cfg = yaml.safe_load(Path(serving_path).read_text(encoding="utf-8"))
        self.point = yaml.safe_load(Path(point_path).read_text(encoding="utf-8"))
        self.limits = self.cfg.get("limits", {})
        self.validated_tags = list(self.cfg.get("validated_tags", []))
        self.weak_tag_prob = float(
            self.cfg.get("suspicious", {}).get("weak_tag_prob", 0.5))
        self.loaded: dict = {}
        self.fallback_reason: str | None = None
        self._keyword_lists = self._load_keywords()
        self._load_models()

    # -- loading -------------------------------------------------------
    def _load_keywords(self) -> dict:
        path = PROJECT_ROOT / "configs" / "keywords.yaml"
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {
            "strong": [s.lower() for s in data.get("strong_scam_keywords", [])],
            "weak": [s.lower() for s in data.get("weak_risk_keywords", [])],
            "veto": [s.lower() for s in data.get("safe_veto_phrases", [])],
        }

    def _load_models(self) -> None:
        import joblib

        wanted = str(self.cfg.get("model", "tfidf"))
        spec = self.cfg["models"][wanted]
        if wanted == "muril":
            run_dir = PROJECT_ROOT / spec["run_dir"]
            if not (run_dir / "config_used.yaml").exists():
                # Ship the model we have, and say why in /health.
                self.fallback_reason = (
                    f"{spec['run_dir']} is missing (no GPU training run on this "
                    "machine), so the service fell back to the TF-IDF model.")
                wanted = "tfidf"
                spec = self.cfg["models"][wanted]
        self.served_model = wanted
        self.spec = spec
        self.verdict_bundle = joblib.load(PROJECT_ROOT / spec["verdict_bundle"])
        tactics_path = PROJECT_ROOT / spec["tactics_bundle"]
        self.tactics_bundle = (joblib.load(tactics_path)
                               if tactics_path.exists() else None)
        self.temperature = float(self.point["calibration"]["temperature"])
        self.modes = {
            name: {
                "threshold": float(info["threshold"]),
                "rule": info.get("rule", ""),
                "blurb": info.get("blurb", ""),
                "val_recall_weighted": info.get("val_recall_weighted"),
                "val_fpr_weighted": info.get("val_fpr_weighted"),
            }
            for name, info in self.point.get("modes", {}).items()
        }
        if not self.modes:
            raise SystemExit("operating_point.yaml has no modes - run "
                             "python -m src.calibrate first")
        self.default_mode = str(self.cfg.get("default_mode", "balanced"))
        if self.default_mode not in self.modes:
            self.default_mode = next(iter(self.modes))

    # -- info ----------------------------------------------------------
    def health(self) -> dict:
        return {
            "status": "ok",
            "serving_model": self.served_model,
            "model_name": self.spec.get("display_name", self.served_model),
            "fallback": self.fallback_reason,
            "modes": list(self.modes),
            "default_mode": self.default_mode,
            "temperature": self.temperature,
            "tactics_head": self.tactics_bundle is not None,
            "validated_tags": self.validated_tags,
            "validated_on": self.spec.get("validated_on"),
            "limits": self.limits,
        }

    def mode_table(self) -> list[dict]:
        """The three operating points with their measured trade-off."""
        from src.phase4_common import rate_with_ci  # local: keep import cheap

        measured = self.point.get("mode_measured") or {}
        rows = []
        for name, info in self.modes.items():
            row = {"mode": name, **info}
            row.update(measured.get(name, {}))
            rows.append(row)
        return rows

    # -- validation ----------------------------------------------------
    def validate(self, text: str) -> str:
        if text is None or not str(text).strip():
            raise ValueError("text must not be empty")
        clean = sanitize_text(text)
        lo = int(self.limits.get("min_chars", 1))
        hi = int(self.limits.get("max_chars", 1000))
        if len(clean) < lo:
            raise ValueError(f"text is too short after cleaning (min {lo} chars)")
        if len(clean) > hi:
            raise ValueError(f"text is too long ({len(clean)} chars, max {hi})")
        return clean

    # -- scoring -------------------------------------------------------
    def _raw_prob(self, text: str) -> float:
        from src.baselines import predict_scores

        meta = {"url_type": self._url_type(text), "had_phone_number": False,
                "script_style": "english", "source": "api"}
        import pandas as pd

        values = predict_scores(self.verdict_bundle, [text], pd.DataFrame([meta]))
        return float(values[0]) if values else 0.0

    def _url_type(self, text: str) -> str:
        from src.scrub import classify_url

        found = URL_RE.findall(text)
        return classify_url(found[0]) if found else "none"

    def _keyword_reasons(self, text: str) -> list[str]:
        low = text.lower()
        hits = [k for k in self._keyword_lists.get("strong", []) if k in low][:3]
        weak = [k for k in self._keyword_lists.get("weak", []) if k in low][:2]
        veto = [k for k in self._keyword_lists.get("veto", []) if k in low][:2]
        out = []
        if hits:
            out.append("keyword rule hit: " + ", ".join(f"'{h}'" for h in hits))
        if weak:
            out.append("weak risk wording: " + ", ".join(f"'{w}'" for w in weak))
        if veto:
            out.append("legitimate-pattern wording: " + ", ".join(f"'{v}'" for v in veto))
        return out

    def _tactics(self, text: str) -> list[dict]:
        if self.tactics_bundle is None:
            return []
        from src.baselines import predict_tactic_probs

        probs = predict_tactic_probs(self.tactics_bundle, [text])
        rows = [{"tag": tag, "prob": round(float(p[0]), 4),
                 "validated": tag in self.validated_tags}
                for tag, p in probs.items() if p and p[0] >= 0.15]
        rows.sort(key=lambda r: -r["prob"])
        return rows[:5]

    def _highlights(self, text: str, baseline: float, cap: int) -> list[dict]:
        """Word-level occlusion: drop a word, re-score, keep the biggest drops.

        Cost is one model call per word, so the candidate list is capped
        (default 40). The cap is a latency control, NOT a claim that only 40
        words matter - the response says so in `warnings`.
        """
        seen: list[str] = []
        for match in WORD_RE.finditer(text):
            word = match.group(0)
            if word.lower() not in {w.lower() for w in seen}:
                seen.append(word)
        candidates = seen[:cap]
        out = []
        for word in candidates:
            stripped = re.sub(rf"\b{re.escape(word)}\b", " ", text, count=1)
            if stripped.strip() == text.strip():
                continue
            delta = baseline - self._raw_prob(stripped)
            # Use a SMALL threshold. On a confident message (p ~ 1.0) dropping
            # any single word barely moves the score, so a 0.01 cut made the
            # highlight list empty exactly when the user most wants to see
            # evidence. Anything that moves the score at all is kept, then we
            # show the top 10.
            if delta > 0.0005:
                out.append({"word": word, "weight": round(float(delta), 4)})
        if not out:
            # Still nothing? Show the relative impact instead of nothing at
            # all: which words, when removed, lower the score the most.
            deltas = []
            for word in candidates:
                stripped = re.sub(rf"\b{re.escape(word)}\b", " ", text, count=1)
                if stripped.strip() == text.strip():
                    continue
                deltas.append((baseline - self._raw_prob(stripped), word))
            deltas.sort(key=lambda kv: -kv[0])
            out = [{"word": w, "weight": round(float(d), 4)}
                   for d, w in deltas[:3] if d > 0]
        out.sort(key=lambda r: -r["weight"])
        return out[:10]

    # -- the public call -----------------------------------------------
    def analyze(self, text: str, mode: str | None = None) -> dict:
        started = time.perf_counter()
        clean = self.validate(text)
        mode = mode or self.default_mode
        if mode not in self.modes:
            raise ValueError(f"unknown mode {mode!r}; expected one of "
                             f"{sorted(self.modes)}")
        threshold = self.modes[mode]["threshold"]

        raw = self._raw_prob(clean)
        calibrated = float(apply_temperature(np.array([raw]), self.temperature)[0])

        tactics = self._tactics(clean)
        url_type = self._url_type(clean)
        evidence = rule_evidence(
            [clean],
            {t["tag"]: [t["prob"]] for t in tactics},
            url_types=[url_type],
            keyword_hits=[bool(self._keyword_reasons(clean)[:1]
                               and "keyword rule hit" in self._keyword_reasons(clean)[0])])
        verdict = str(decide(np.array([calibrated]), evidence, threshold)[0])

        reasons = []
        if verdict == "suspicious":
            reasons.append("model score clears the threshold but no named "
                           "evidence supports it (no tactic above "
                           f"{self.weak_tag_prob:.2f}, no OTP/payment wording, "
                           "no risky link)")
        if tactics:
            top = tactics[0]
            reasons.append(f"tactic signal: {top['tag']} ({top['prob']:.2f}, "
                           + ("validated" if top["validated"] else "unvalidated")
                           + ")")
        reasons += self._keyword_reasons(clean)
        if url_type not in ("none", ""):
            reasons.append(f"link type: {url_type}")
        reasons.append(f"mode: {mode} (threshold {threshold:.3f})")

        warnings = ["validated on English real messages only; Hinglish is "
                    "synthetic-only evidence"]
        if any(not t["validated"] for t in tactics):
            warnings.append("one or more tactic tags are unvalidated "
                            "(fewer than 20 real positives) - do not trust their precision")
        if verdict == "scam" and calibrated < 0.5:
            warnings.append("verdict clears the threshold but the calibrated "
                            "probability is low: this scorer is bimodal, so the "
                            "number is not a confidence statement")
        if self.fallback_reason:
            warnings.append(self.fallback_reason)
        warnings.append("research prototype - not a substitute for checking with "
                        "your bank")

        highlights = self._highlights(clean, raw,
                                      int(self.limits.get("max_occlusion_words", 40)))
        if len(WORD_RE.findall(clean)) > int(self.limits.get("max_occlusion_words", 40)):
            warnings.append("highlights explain at most the first "
                            f"{self.limits.get('max_occlusion_words', 40)} distinct words")

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "verdict": verdict,
            "scam_probability": round(calibrated, 6),
            "scam_probability_raw": round(raw, 6),
            "mode": mode,
            "threshold": round(threshold, 6),
            "tactics": tactics,
            "reasons": reasons,
            "highlights": highlights,
            "warnings": warnings,
            "model_info": {
                "name": self.spec.get("display_name", self.served_model),
                "key": self.served_model,
                "mode": mode,
                "training_mix": self.spec.get("training_mix"),
                "validated_on": self.spec.get("validated_on"),
                "temperature": self.temperature,
            },
            "latency_ms": round(elapsed_ms, 1),
        }


_REGISTRY: Registry | None = None


def get_registry() -> Registry:
    """Lazy singleton: load once per process, reuse across requests."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
    return _REGISTRY


def reset_registry() -> None:
    """Test hook: forget the cached registry."""
    global _REGISTRY
    _REGISTRY = None


if __name__ == "__main__":
    reg = Registry()
    print(json.dumps(reg.health(), indent=2))
    demo = reg.analyze("Your account will be blocked today. Share the OTP now.",
                       mode=reg.default_mode)
    print(json.dumps({k: demo[k] for k in ("verdict", "scam_probability",
                                          "tactics", "warnings", "latency_ms")},
                     indent=2))
