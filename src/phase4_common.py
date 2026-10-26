"""
src/phase4_common.py - the SHARED plumbing of Phase 4.

WHY IT EXISTS
  Every Phase 4 script (error analysis, novelty buckets, calibration,
  abstention, robustness probes) must work for ANY model:
      python -m src.error_analysis --model verdict_lr_plus
      python -m src.calibrate      --model e1_muril
  so the model-agnostic bits live here, once:
    * a Predictor that either READS outputs/<name>/preds_<split>.parquet
      (a transformer run that already predicted) or SCORES text on the fly
      with a TF-IDF baseline from models/. Same call either way.
    * calibration maths (temperature scaling, ECE, reliability bins) that is
      fitted on VALIDATION ONLY - assert_val_only() enforces that in code.
    * the abstention operating point (t_high / t_low) with the evidence
      (how many safe rows stand behind each threshold) printed, never hidden.

THE RULES THIS FILE FOLLOWS
  * calibration and thresholds never read a test split (hard check below);
  * probabilities are RAW model scores until a temperature is applied; both
    are always kept as separate columns;
  * a stratum with fewer than MIN_ROWS_FOR_CLAIMS rows is flagged
    "too small for claims" instead of being quoted as a number;
  * probes (source=probe_*) are scored here but never enter messages.parquet,
    so they can never be trained on.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, SEED, data_path
from src.io_utils import load_messages

# Same gate src/evaluate.py uses: smaller than this = too small for claims.
MIN_ROWS_FOR_CLAIMS = 30
# Splits that may be used to FIT anything. Everything else is for reporting.
VAL_SPLITS = {"val"}
TEST_SPLITS = {"test_a", "test_b", "test_real", "suspicious_holdout", "probe"}
# Probe rows carry source=probe_* and are NEVER in messages.parquet.
PROBE_SOURCE_PREFIX = "probe_"
# Reliability-diagram bins (equal-width on probability).
N_RELIABILITY_BINS = 10
# Probability floor so log-odds never explodes on a 0/1 prediction.
EPS = 1e-6


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def too_small(n: int) -> bool:
    return int(n) < MIN_ROWS_FOR_CLAIMS


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval for k successes out of n.

    Wilson, not normal-approximation: at n=59 (real ham) and around p=0.05 the
    textbook interval is nonsense, and at n=0 it would divide by zero. This is
    the interval every rate in the Phase 5 report carries.
    """
    n = int(n)
    if n <= 0:
        return None, None
    k = int(k)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate_with_ci(k: int, n: int, digits: int = 3) -> str:
    """'0.153 (0.085-0.251), n=59' - the format every table uses."""
    if int(n) <= 0:
        return "n/a (n=0)"
    lo, hi = wilson_ci(k, n)
    return (f"{k / n:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f}), n={int(n)}")


def weighted_rate(y, p, w, threshold: float, positive: bool = True) -> float | None:
    """Weighted share of the chosen class above/below a threshold.

    positive=True  -> share of the positive class with p >= threshold (recall)
    positive=False -> share of the negative class with p >= threshold (FPR)
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
    mask = (y == 1) if positive else (y == 0)
    total = float(w[mask].sum())
    if mask.sum() == 0 or total <= 0:
        return None
    return float((w[mask] * (p[mask] >= threshold)).sum() / total)


def claim_flag(n: int) -> str:
    """The note every table puts next to a too-small n."""
    return f"**too small for claims** (n={int(n)} < {MIN_ROWS_FOR_CLAIMS})" if too_small(n) else ""


def is_probe(df: pd.DataFrame) -> pd.Series:
    if "source" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["source"].astype(str).str.startswith(PROBE_SOURCE_PREFIX)


def assert_val_only(frame: pd.DataFrame, what: str) -> None:
    """Hard guard: nothing may be FITTED on a test split.

    Calibration and threshold selection are validation-only decisions. This
    check makes it impossible for a future edit to quietly refit them on a
    test set - it raises instead.
    """
    splits = set(frame["split"].astype(str)) if "split" in frame.columns else set()
    bad = splits - VAL_SPLITS
    if bad:
        raise AssertionError(
            f"refusing to FIT {what} on non-validation rows: {sorted(bad)}. "
            "Temperature and thresholds are validation-only (Phase 4 rule).")


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def split_frame(split: str) -> pd.DataFrame:
    """One split of the main table with the columns Phase 4 always uses."""
    df = load_messages()
    sub = df[df["split"] == split].reset_index(drop=True)
    if sub.empty:
        return sub
    if "real_stratum" not in sub.columns:
        sub["real_stratum"] = ""
    sub["real_stratum"] = sub["real_stratum"].fillna("").astype(str)
    return sub


def evaluation_targets() -> list[tuple[str, pd.DataFrame]]:
    """The (name, frame) pairs every Phase 4 table reports.

    Real strata are separate because a blended real number is not a result;
    the two synthetic dev splits are kept for comparison.
    """
    out: list[tuple[str, pd.DataFrame]] = []
    val = split_frame("val")
    if not val.empty:
        for part, mask in (
            ("val_synthetic", val["source"].isin(["synthetic", "synthetic_llm"])),
            ("val_real", ~val["source"].isin(["synthetic", "synthetic_llm"])),
        ):
            out.append((part, val[mask].reset_index(drop=True)))
    for split in ("test_b", "test_real"):
        frame = split_frame(split)
        if frame.empty:
            continue
        if split == "test_b":
            out.append(("test_b", frame))
            continue
        for strat in ("user_real_scam", "user_real_safe", "external_scam",
                      "public_ham_english"):
            sub = frame[frame["real_stratum"] == strat].reset_index(drop=True)
            if not sub.empty:
                out.append((f"test_real/{strat}", sub))
    return out


def probes_frame() -> pd.DataFrame:
    """Every robustness probe written by src/robustness.py (empty if none)."""
    path = data_path("probes_parquet")
    if not path.exists():
        return pd.DataFrame()
    df = load_messages(path)
    df["probe_kind"] = df["source"].astype(str).str[len(PROBE_SOURCE_PREFIX):]
    return df


# ---------------------------------------------------------------------------
# predictors - model-agnostic
# ---------------------------------------------------------------------------
class BaselinePredictor:
    """A TF-IDF / rules baseline from models/*.joblib, scored on the fly."""

    kind = "baseline"
    can_score = True

    def __init__(self, name: str, bundle: dict):
        self.name = name
        self.bundle = bundle

    def scores(self, frame: pd.DataFrame) -> np.ndarray | None:
        from src.baselines import predict_scores

        values = predict_scores(self.bundle, frame["text"].astype(str).tolist(), frame)
        if values is None:
            raise SystemExit(
                f"{self.name} has no probability output (majority/rules); "
                "Phase 4 needs a model that can rank. Use verdict_lr_plus.")
        return np.asarray(values, dtype=float)


class TransformerPredictor:
    """A trained Phase 3 run from outputs/<name>/, scored on the fly.

    Used for probes (perturbed / translated text), which are not in any
    preds_*.parquet. For ordinary splits we prefer the saved predictions.
    """

    kind = "transformer"
    can_score = True

    def __init__(self, run_name: str, output_dir: str = "outputs",
                 batch_size: int = 64):
        import torch
        import yaml
        from transformers import AutoTokenizer

        from src.data_torch import VerdictTagDataset, load_label_maps, make_collate_fn
        from src.model import MultiTaskClassifier

        self._torch = torch
        self._Dataset = VerdictTagDataset
        self._collate = make_collate_fn
        self.batch_size = batch_size

        run_dir = Path(PROJECT_ROOT) / output_dir / run_name
        if not (run_dir / "config_used.yaml").exists():
            raise SystemExit(
                f"no run {run_name!r} in {run_dir}. Train it first "
                "(python -m src.run_experiments) or pass --model verdict_lr_plus.")
        self.run_dir = run_dir
        self.name = run_name
        cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
        self.maps = load_label_maps(run_dir / "label_maps.json")
        self.max_length = int(cfg.get("max_length", 128))
        self.cfg = cfg
        tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTaskClassifier(cfg["model_name"], self.maps["num_verdicts"],
                                    self.maps["num_tags"])
        state = torch.load(run_dir / cfg.get("checkpoint_name", "best_model.pt"),
                           map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model

    def scores(self, frame: pd.DataFrame) -> np.ndarray | None:
        torch = self._torch
        classes = list(self.maps["verdict_classes"])
        if "scam" not in classes:
            return None
        idx = classes.index("scam")
        work = frame.copy()
        # VerdictTagDataset needs a verdict column; probes carry a gold label,
        # but at scoring time we only ever read the PROBABILITY from it.
        if "tactics" not in work.columns:
            work["tactics"] = [[]] * len(work)
        if "verdict" not in work.columns:
            work["verdict"] = "safe"
        else:
            # `suspicious` is not a class. These rows are scored for their
            # probability only, so a placeholder label is safe - but it must
            # never leak back into any metric.
            known = set(self.maps["verdict_to_id"])
            work["verdict"] = work["verdict"].astype(str).where(
                work["verdict"].astype(str).isin(known), "safe")
        ds = self._Dataset(work, self.tokenizer, self.maps,
                           max_length=self.max_length)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=self._collate(self.tokenizer.pad_token_id))
        out = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                v_logits, _ = self.model(batch["input_ids"], batch["attention_mask"])
                out.append(torch.softmax(v_logits, dim=-1)[:, idx].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=float)


def load_predictor(name: str, output_dir: str = "outputs"):
    """Return the Predictor for `name`.

    Order: a saved transformer run in outputs/ wins, else a TF-IDF baseline
    in models/. The name is what every Phase 4 script passes as --model.
    """
    run_dir = Path(PROJECT_ROOT) / output_dir / name
    if (run_dir / "config_used.yaml").exists():
        print(f"  predictor: transformer run {name!r} ({run_dir})")
        return TransformerPredictor(name, output_dir=output_dir)
    from src.baselines import MODEL_FILES

    if name in MODEL_FILES:
        import joblib

        from src.config import data_path as _data_path
        path = Path(PROJECT_ROOT) / "models" / MODEL_FILES[name]
        if not path.exists():
            raise SystemExit(f"{path} missing - run: python -m src.baselines")
        print(f"  predictor: TF-IDF baseline {name!r} ({path.name})")
        return BaselinePredictor(name, joblib.load(path))
    raise SystemExit(
        f"unknown --model {name!r}: expected a run name in {run_dir.parent} or "
        "a baseline name like verdict_lr_plus.")


def saved_predictions(run_name: str, split: str) -> pd.DataFrame | None:
    """outputs/<run>/preds_<split>.parquet if a transformer run produced it."""
    path = Path(PROJECT_ROOT) / "outputs" / run_name / f"preds_{split}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return df


def score_frame(predictor, frame: pd.DataFrame) -> np.ndarray:
    """P(scam) for every row of `frame`, for any predictor."""
    if frame.empty:
        return np.zeros(0, dtype=float)
    values = predictor.scores(frame)
    if values is None:
        raise SystemExit(f"{getattr(predictor, 'name', '?')} cannot produce scores")
    return np.asarray(values, dtype=float)


def predictions_for_split(predictor, name: str, split: str) -> pd.DataFrame:
    """Truth + RAW probability for one split, model-agnostically.

    A transformer run's saved preds_*.parquet is preferred (it is exactly what
    src/evaluate.py scored); otherwise the baseline is run on the fly.
    """
    cached = saved_predictions(name, split) if predictor.kind == "transformer" else None
    if cached is not None:
        out = cached.copy()
        if "prob_scam" not in out.columns:
            raise SystemExit(f"{name}/preds_{split}.parquet has no prob_scam column")
        out["prob_raw"] = out["prob_scam"].astype(float)
        if "real_stratum" not in out.columns:
            out["real_stratum"] = ""
        out["real_stratum"] = out["real_stratum"].fillna("").astype(str)
        out["from_saved_preds"] = True
        return out.reset_index(drop=True)

    frame = split_frame(split).copy()
    frame["prob_raw"] = score_frame(predictor, frame)
    frame["from_saved_preds"] = False
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# calibration maths (validation-only by construction)
# ---------------------------------------------------------------------------
def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def apply_temperature(p, temperature: float) -> np.ndarray:
    """Rescale raw probabilities with a learned temperature (logit space)."""
    return sigmoid(logit(p) / float(temperature))


def nll(y, p, weights=None) -> float:
    p = _clip(p)
    terms = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if weights is None:
        return float(terms.mean())
    w = np.asarray(weights, dtype=float)
    return float((terms * w).sum() / w.sum())


def fit_temperature(y, p, weights=None) -> float:
    """One-parameter temperature scaling, minimising weighted NLL.

    Golden-section search on log(T) - no optimiser, so the result is
    deterministic given the same validation rows (seed-independent maths).
    T=1.0 means "already calibrated"; T>1 means over-confident.
    """
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    lo, hi = math.log(0.05), math.log(20.0)
    gr = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(log_t: float) -> float:
        return nll(y, apply_temperature(p, math.exp(log_t)), weights)

    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(80):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = objective(d)
    return float(math.exp((a + b) / 2.0))


def reliability_bins(y, p, weights=None,
                     bins: int = N_RELIABILITY_BINS) -> list[dict]:
    """Equal-width probability bins: n, mean prediction, empirical rate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        weight = float(w[mask].sum())
        out.append({
            "bin": i,
            "p_lo": float(lo), "p_hi": float(hi),
            "n": int(mask.sum()),
            "weight": weight,
            "mean_pred": float(p[mask].mean()) if mask.any() else None,
            "empirical_rate": (float((y[mask] * w[mask]).sum() / weight)
                               if weight > 0 else None),
        })
    return out


def ece(y, p, weights=None, bins: int = N_RELIABILITY_BINS) -> float | None:
    """Expected calibration error, weighted if requested."""
    rows = reliability_bins(y, p, weights, bins)
    total = sum(r["weight"] for r in rows)
    if total <= 0:
        return None
    value = 0.0
    for r in rows:
        if r["weight"] <= 0 or r["mean_pred"] is None or r["empirical_rate"] is None:
            continue
        value += (r["weight"] / total) * abs(r["mean_pred"] - r["empirical_rate"])
    return float(value)


def equal_cell_weights(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Equal weight per (real|synthetic) x (scam|safe) cell - Phase 4 amendment.

    WHY: the validation split is 98.3% scam and ~87% real. Fitting on raw val
    therefore optimises for "flag real-looking text as scam". Weighing each of
    the four cells equally makes the 60 real ham rows count as much as the
    3,511 real scams, so both threshold decisions have real evidence.

    Returns (weights, cell_counts) - the counts are printed next to every
    threshold so nobody can miss how thin the safe evidence is.
    """
    is_real = ~frame["source"].isin(["synthetic", "synthetic_llm"])
    is_scam = frame["verdict"].astype(str) == "scam"
    cell = np.where(is_real, "real", "synthetic") + "/" + np.where(is_scam, "scam", "safe")
    counts = {str(k): int(v) for k, v in pd.Series(cell).value_counts().items()}
    weights = np.zeros(len(frame), dtype=float)
    for name in counts:
        mask = cell == name
        weights[mask] = 1.0 / (len(counts) * counts[str(name)])
    return weights, counts


# ---------------------------------------------------------------------------
# abstention operating point
# ---------------------------------------------------------------------------
def fit_thresholds(y, p, weights=None,
                   min_precision: float = 0.95, min_npv: float = 0.95) -> dict:
    """t_high / t_low from VALIDATION rows only.

    t_high = smallest probability where "predict scam" keeps precision >= 0.95.
    t_low  = largest probability where "predict safe" keeps NPV      >= 0.95.
    Everything between is the suspicious (abstain) band.

    The caller passes validation data only - assert_val_only() is called by
    src/calibrate.py before this, and the evidence (how many rows are behind
    each decision) is returned so it can be printed.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(-p, kind="stable")
    ys, ws = y[order], w[order]
    cum_n = np.cumsum(ws)
    cum_pos = np.cumsum(ys * ws)
    precision = np.divide(cum_pos, cum_n, out=np.zeros_like(cum_pos), where=cum_n > 0)

    t_high, prec_at_high, n_above = None, None, 0
    hits = np.flatnonzero(precision >= min_precision)
    if len(hits):
        k = int(hits[-1])              # smallest threshold that still clears the bar
        t_high = float(p[order][k])
        prec_at_high = float(precision[k])
        n_above = int((p >= t_high).sum())

    order_lo = np.argsort(p, kind="stable")
    ys_lo, ws_lo = y[order_lo], w[order_lo]
    cum_n_lo = np.cumsum(ws_lo)
    cum_safe = np.cumsum((1.0 - ys_lo) * ws_lo)
    npv = np.divide(cum_safe, cum_n_lo, out=np.zeros_like(cum_safe), where=cum_n_lo > 0)
    t_low, npv_at_low, n_below = None, None, 0
    hits_lo = np.flatnonzero(npv >= min_npv)
    if len(hits_lo):
        k = int(hits_lo[-1])           # largest threshold that still clears the bar
        t_low = float(p[order_lo][k])
        npv_at_low = float(npv[k])
        n_below = int((p <= t_low).sum())

    # How many validation rows actually sit INSIDE the band (t_low, t_high)?
    # A band with no rows cannot abstain on anything, so the operating point
    # collapses to a single cut - that is a fact about the model's scores on
    # validation, and the report says so instead of hiding it.
    n_between = 0
    if t_high is not None and t_low is not None:
        n_between = int(((p > t_low) & (p < t_high)).sum())

    conflict = False
    band_reason = "ok"
    if t_high is not None and t_low is not None and t_low >= t_high:
        # The two decisions overlap: they meet (or cross). With NO validation
        # rows between them the honest reading is "a single cut satisfies both
        # targets", so t_low = t_high and the band is declared unavailable.
        conflict = True
        if n_between == 0:
            band_reason = ("no validation rows between t_low and t_high: "
                           "a single cut satisfies both targets, so the "
                           "abstain band is empty by construction")
        else:
            band_reason = ("targets cross: the safe-side bar sits above the "
                           "scam-side bar, so the pair was collapsed")
        t_low = float(t_high)

    return {
        "n_rows_in_band": n_between,
        "abstention_available": bool(n_between > 0 and not conflict),
        "band_reason": band_reason,
        "t_high": t_high, "t_low": t_low,
        "min_precision_target": min_precision,
        "min_npv_target": min_npv,
        "precision_at_t_high": prec_at_high,
        "npv_at_t_low": npv_at_low,
        "n_rows": int(len(y)),
        "n_scam": int((y == 1).sum()),
        "n_safe": int((y == 0).sum()),
        "n_definitive_scam": n_above,
        "n_definitive_safe": n_below,
        # The evidence behind each decision - printed, never hidden.
        "n_safe_behind_t_low": int(((y == 0) & (p <= (t_low if t_low is not None else -1))).sum()),
        "n_scam_behind_t_high": int(((y == 1) & (p >= (t_high if t_high is not None else 2))).sum()),
        "targets_achievable": {"t_high": t_high is not None, "t_low": t_low is not None},
        "bands_overlap": conflict,
    }


def apply_band(p, t_high: float | None, t_low: float | None) -> np.ndarray:
    """scam / suspicious / safe verdict from two thresholds.

    `safe` is written first so that a row sitting EXACTLY on a shared cut is
    called scam, not safe: equal to `t_high` means "the scam bar was met".
    """
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), "suspicious", dtype=object)
    if t_low is not None:
        out[p <= t_low] = "safe"
    if t_high is not None:
        out[p >= t_high] = "scam"
    return out


def band_summary(frame: pd.DataFrame, p, t_high, t_low,
                 truth: np.ndarray | None = None) -> dict:
    """Coverage / accuracy of the three-way decision on one frame."""
    labels = apply_band(p, t_high, t_low)
    definitive = labels != "suspicious"
    summary = {
        "n_rows": int(len(frame)),
        "n_scam": int((labels == "scam").sum()),
        "n_suspicious": int((labels == "suspicious").sum()),
        "n_safe": int((labels == "safe").sum()),
        "coverage": float(definitive.mean()) if len(frame) else None,
    }
    if truth is not None and definitive.any():
        correct = (labels[definitive] == truth[definitive]).astype(float)
        summary["accuracy_on_covered"] = float(correct.mean())
        summary["n_covered"] = int(definitive.sum())
        summary["too_small_for_claims"] = too_small(int(definitive.sum()))
    else:
        summary["accuracy_on_covered"] = None
        summary["n_covered"] = int(definitive.sum())
        summary["too_small_for_claims"] = too_small(int(definitive.sum()))
    summary["too_small_total"] = too_small(len(frame))
    return summary
