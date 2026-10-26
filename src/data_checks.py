"""
src/data_checks.py - validation + a plain-text data quality report.

Two jobs:
  1. validate_schema(df)  - the CONTRACT for data/processed/messages.parquet.
     Fails loudly on missing columns, bad enum values, null/empty text,
     empty verdict, malformed dates, or wrong tactic tags.
  2. data_report(df)      - counts you actually want to see before modeling:
     verdict x source, tactic frequency, script_style, url_type, length
     stats, cluster stats, split balance, leakage check PASS/FAIL.
     Written to data/processed/data_report.txt and printed.

Run:  python -m src.data_checks
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from src.config import (
    BINARY_VERDICTS,
    CFG,
    CHANNELS,
    CONFIDENCE_LEVELS,
    MESSAGE_COLUMNS,
    REAL_STRATA,
    SCRIPT_STYLES,
    SOURCES,
    SPLITS,
    TACTIC_TAGS,
    URL_TYPES,
    VERDICTS,
    data_path,
)

# Date strings we accept: YYYY-MM-DD or YYYY-MM (nothing else).
DATE_OK_RE = __import__("re").compile(r"^\d{4}-\d{2}(-\d{2})?$")

# ---------------------------------------------------------------------------
# TEXT NORMALISATION (Phase 2.5 duplicate control)
# ---------------------------------------------------------------------------
def normalize_text(value) -> str:
    """A canonical form used to decide whether two rows say the SAME thing.

    Rules (exactly what the Phase 2.5 spec asks for):
      * lowercase
      * collapse every run of whitespace into one space
      * replace every run of digits with "0" (a date, a rupee amount or a
        reference number changing must NOT hide a duplicate)
    Without this, "Rs 500 OTP!" and "rs   500 otp !" look different.
    """
    s = "" if value is None else str(value).lower()
    s = __import__("re").sub(r"\d+", "0", s)          # 50,000 / 50000 -> 0
    s = __import__("re").sub(r"\s+", " ", s)          # collapse whitespace
    return s.strip()


def exact_text_key(value) -> str:
    """A STRICTER key for split-leakage checks: case + whitespace only.

    `normalize_text` also zeroes digits, which is right for near-duplicate
    *counting* but too aggressive for leakage: "pay Rs 500" and "pay Rs 900"
    are different messages, not the same row copied twice. For leakage we only
    treat text as identical when it is the same sentence modulo casing and
    spacing, so a real copy-paste can never hide behind a number.
    """
    s = "" if value is None else str(value).lower()
    s = __import__("re").sub(r"\s+", " ", s)          # collapse whitespace
    return s.strip()



def as_tactics_list(value) -> list[str]:
    """Accept every shape the tactics column can arrive in:

    * a real list / tuple            (in-memory DataFrame)
    * a numpy array                  (AFTER a parquet round-trip!)
    * 'a;b;c'                        (label_cli writes CSV this way)
    * "['a', 'b']" or '[]'           (synthetic.csv holds list reprs)
    * empty / None / NaN             -> []
    """
    if value is None:
        return []
    # Parquet stores lists as numpy arrays - any non-string iterable is
    # already a ready-made sequence of tags.
    if not isinstance(value, str) and hasattr(value, "__iter__"):
        return [str(v) for v in value]
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return [p.strip() for p in s.split(";") if p.strip()]


def validate_schema(df: pd.DataFrame, strict_empty_ok: bool = False) -> None:
    """Raise ValueError with a clear message if the DataFrame breaks the contract.

    This is what protects every later phase from typo'd verdicts, missing
    columns or leaked text. It is intentionally LOUD - a silent bad dataset
    is far worse than a crash here.
    """
    problems: list[str] = []

    # 1. Missing columns -----------------------------------------------------
    missing = [c for c in MESSAGE_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    if not problems:  # only check values if the columns exist
        # 2. Empty text / verdict -------------------------------------------
        if df["text"].isna().any():
            n = int(df["text"].isna().sum())
            problems.append(f"{n} row(s) have null text")
        blank_text = df["text"].astype(str).str.strip() == ""
        if blank_text.any():
            problems.append(f"{int(blank_text.sum())} row(s) have empty text")
        if df["verdict"].isna().any() or (df["verdict"].astype(str).str.strip() == "").any():
            problems.append("some row(s) have an empty verdict - every row needs "
                            "scam/suspicious/safe")

        # 3. Enum columns ----------------------------------------------------
        enum_checks = {
            "verdict": VERDICTS,
            "channel": CHANNELS,
            "script_style": SCRIPT_STYLES,
            "source": SOURCES,
            "label_confidence": CONFIDENCE_LEVELS,
            "url_type": URL_TYPES,
            "split": SPLITS,
            # "" = the row is not real, so it has no real stratum.
            "real_stratum": tuple(list(REAL_STRATA) + [""]),
        }
        for col, allowed in enum_checks.items():
            bad = sorted(set(df[col].astype(str)) - set(allowed))
            if bad:
                problems.append(f"bad {col} value(s) {bad}; allowed: {list(allowed)}")

        # 4. tactics: every tag must be in the taxonomy -----------------------
        for idx, val in df["tactics"].items():
            for tag in as_tactics_list(val):
                if tag not in TACTIC_TAGS:
                    problems.append(f"row {idx}: unknown tactic tag {tag!r}")
                    break

        # 5. date format -----------------------------------------------------
        for idx, val in df["date_received"].items():
            s = "" if val is None else str(val).strip()
            if s in {"", "nan", "None", "NaT"}:
                continue  # unknown date is allowed
            if not DATE_OK_RE.match(s):
                problems.append(f"row {idx}: bad date_received {val!r} "
                                f"(want YYYY-MM-DD, YYYY-MM or empty)")
                break

        # 6. dtypes / basics -------------------------------------------------
        # template_family: every row needs a string (use "" when there is no
        # family). NaN would silently become the literal text "nan" later.
        if "template_family" in df.columns and df["template_family"].isna().any():
            n = int(df["template_family"].isna().sum())
            problems.append(
                f"{n} row(s) have a null template_family - use \"\" when a "
                "message belongs to no template family"
            )
        if not df["had_phone_number"].map(lambda x: isinstance(x, (bool,))).all():
            # CSV-loaded frames may carry 'True'/'False' strings - only warn.
            if not df["had_phone_number"].astype(str).str.lower().isin(["true", "false"]).all():
                problems.append("had_phone_number must be boolean")
        tc = pd.to_numeric(df["template_cluster"], errors="coerce")
        if tc.isna().any():
            problems.append("template_cluster must be integer for every row")
        if df["id"].duplicated().any():
            dups = df.loc[df["id"].duplicated(), "id"].head(5).tolist()
            problems.append(f"duplicate id(s): {dups}")

    if problems:
        raise ValueError("SCHEMA VALIDATION FAILED:\n  - " + "\n  - ".join(problems))
    print(f"validate_schema: OK ({len(df)} rows, {len(MESSAGE_COLUMNS)} columns)")


def leakage_check(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Return (PASS/FAIL, list of problems) for the Phase 2 rules:

      * no template_cluster in more than one split
      * NO two rows with the SAME normalised text in different splits
        (Phase 2.5: the cluster check alone can miss exact copies)
      * NO synthetic row (source=synthetic/synthetic_llm) in test_real
      * public_weak rows, when present at all, stay in train (ablation build)
      * a non-empty template_family never crosses splits - which includes the
        test_b rule: a held-out family may not appear in train/val/test_a
    """
    problems: list[str] = []

    # (a) one cluster may live in exactly one split
    spans = df.groupby("template_cluster")["split"].nunique()
    bad = spans[spans > 1]
    if len(bad) > 0:
        problems.append(f"{len(bad)} cluster(s) span multiple splits: {bad.index[:10].tolist()}")

    # (a2) EXACT-TEXT leakage (Phase 2.5): the same message may never appear
    # in two different splits, even if clustering missed it. This fails
    # LOUDLY because a memorised test message silently inflates every score.
    norm = df["text"].map(exact_text_key)
    tmp = pd.DataFrame({"norm": norm, "split": df["split"]})
    norm_spans = tmp.groupby("norm")["split"].nunique()
    bad_norm = norm_spans[norm_spans > 1]
    if len(bad_norm) > 0:
        examples = [t[:60] for t in bad_norm.index[:3].tolist()]
        problems.append(f"{len(bad_norm)} identical text(s) span multiple splits "
                        f"(EXACT-TEXT LEAKAGE): {examples}")

    # (b) HARD RULE: machine-generated rows are never "real" test data
    synthetic = df["source"].isin(["synthetic", "synthetic_llm"])
    escaped = df[synthetic & (df["split"] == "test_real")]
    if len(escaped) > 0:
        problems.append(f"{len(escaped)} synthetic row(s) in test_real: "
                        f"{escaped['id'].head(10).tolist()}")

    # (c) English UCI spam (public_weak) is not allowed into any eval split
    pw = df[(df["source"] == "public_weak") & (df["split"] != "train")]
    if len(pw) > 0:
        problems.append(f"{len(pw)} public_weak row(s) outside train: "
                        f"{pw['id'].head(10).tolist()}")

    # (d) one family may live in exactly one split (covers test_b isolation)
    if "template_family" in df.columns:
        fam = df[df["template_family"].astype(str).str.strip() != ""]
        if len(fam) > 0:
            fam_spans = fam.groupby("template_family")["split"].nunique()
            bad_fam = fam_spans[fam_spans > 1]
            if len(bad_fam) > 0:
                problems.append(
                    f"{len(bad_fam)} template_family(s) span multiple splits "
                    f"(test_b isolation broken): {bad_fam.index[:10].tolist()}"
                )
    return ("FAIL" if problems else "PASS"), problems


def binary_table_check(df: pd.DataFrame) -> list[str]:
    """Phase 2.5: the MAIN table must be BINARY (scam/safe only).

    Rows labelled `suspicious` belong in data/processed/suspicious_holdout
    parquet (Phase 4 abstention material), never in the table the models
    train on or are scored against. Returns a list of problems (empty = ok).
    """
    problems: list[str] = []
    counts = df["verdict"].value_counts()
    n_susp = int(counts.get("suspicious", 0))
    if n_susp:
        problems.append(
            f"{n_susp} row(s) still labelled 'suspicious' in the MAIN table - "
            "suspicious is an unsure band, not a class. Move them to "
            "suspicious_holdout.parquet (run python -m src.build_dataset)."
        )
    bad = sorted(set(df["verdict"].astype(str)) - set(BINARY_VERDICTS))
    if bad:
        problems.append(f"unexpected verdict(s) {bad}; binary table allows "
                        f"{list(BINARY_VERDICTS)}")
    return problems


def _distinct_ngrams(texts, n: int = 2) -> int:
    """How many DIFFERENT n-grams a set of texts uses (diversity, not count)."""
    grams: set[str] = set()
    for t in texts:
        words = str(t).lower().split()
        grams.update(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    return len(grams)


def diversity_report(df: pd.DataFrame) -> list[str]:
    """Phase 2.5 diversity metrics: is the corpus repetitive?

    * unique ratio  = rows with a UNIQUE normalised text / all rows.
      Target for synthetic data: >= 95% (configs/config.yaml:
      diversity.min_unique_ratio). Anything lower means the generator just
      repeats the same sentence - the model learns nothing and the numbers
      become fragile.
    * distinct-2gram per family = how many different word pairs a template
      family actually uses. A family scoring 1 is one sentence re-printed.
    """
    lines: list[str] = ["--- unique-text diversity (normalised: lower/collapse/digits->0) ---"]
    min_ratio = float(CFG_DIVERSITY["min_unique_ratio"])

    # The final table has already had exact duplicates removed, so its ratio
    # is always 100% by construction. The HONEST diversity number is measured
    # on the raw generator output BEFORE dedupe - show both.
    raw_path = data_path("synthetic_parquet")
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
        raw_norm = raw["text"].map(normalize_text)
        n, u = len(raw), int(raw_norm.nunique())
        lines.append(f"RAW generator corpus (before dedupe): {u}/{n} unique "
                     f"({u / max(n, 1):.1%}) - target >= {min_ratio:.0%}")
        for verdict, sub in raw.groupby("verdict"):
            sn = sub["text"].map(normalize_text)
            lines.append(f"  {verdict:10s} {sn.nunique():5d}/{len(sub):<5d} "
                         f"({sn.nunique() / max(len(sub), 1):.1%})")
        raw_fam = []
        for fam, sub in raw.groupby(raw["template_family"].astype(str).str.strip()):
            if len(sub) < 2:
                continue
            sn = sub["text"].map(normalize_text)
            g2 = _distinct_ngrams(sub["text"], 2)
            raw_fam.append((sn.nunique() / len(sub), sn.nunique(), len(sub), g2, fam))
        raw_fam.sort()
        lines.append("  worst raw families:")
        for rr, uu, nn, g2, fam in raw_fam[:8]:
            mark = ""
            if rr < min_ratio:
                mark += f"  <-- unique {rr:.0%} (< {min_ratio:.0%})"
            if g2 < int(CFG_DIVERSITY["min_distinct2_per_family"]):
                mark += f"  <-- only {g2} distinct 2-gram(s)"
            lines.append(f"    {fam:32s} {uu:4d}/{nn:<4d} ({rr:.0%}), "
                         f"distinct-2gram={g2}{mark}")
        lines.append("AFTER exact dedupe (the table models actually use):")

    norm = df["text"].map(normalize_text)

    def ratio(mask) -> tuple[int, int, float]:
        sub = norm[mask]
        n = len(sub)
        u = int(sub.nunique())
        return n, u, (u / n if n else 0.0)

    n, u, r = ratio(pd.Series(True, index=df.index))
    lines.append(f"overall: {u}/{n} unique ({r:.1%})")

    lines.append("per source:")
    for src, sub in df.assign(_norm=norm).groupby("source"):
        nn = len(sub)
        uu = int(sub["_norm"].nunique())
        rr = uu / nn if nn else 0.0
        mark = ""
        if str(src).startswith("synthetic") and rr < min_ratio:
            mark = f"  <-- BELOW {min_ratio:.0%}: generator repeats itself"
        lines.append(f"  {src:20s} {uu:5d}/{nn:<5d} ({rr:.1%}){mark}")

    lines.append("per template_family (worst 8):")
    fam_rows = []
    for fam, sub in df.assign(_norm=norm).groupby(df["template_family"].astype(str).str.strip()):
        fam = fam or "(no family)"
        nn = len(sub)
        if nn < 2:
            continue
        uu = int(sub["_norm"].nunique())
        g2 = _distinct_ngrams(sub["text"], 2)
        fam_rows.append((uu / nn, uu, nn, g2, fam))
    fam_rows.sort()          # worst (most repetitive) first
    min_g2 = int(CFG_DIVERSITY["min_distinct2_per_family"])
    for rr, uu, nn, g2, fam in fam_rows[:8]:
        mark = ""
        if rr < min_ratio:
            mark += f"  <-- unique {rr:.0%} (< {min_ratio:.0%})"
        if g2 < min_g2:
            mark += f"  <-- only {g2} distinct 2-gram(s)"
        lines.append(f"  {fam:34s} {uu:4d}/{nn:<4d} ({rr:.0%}), "
                     f"distinct-2gram={g2}{mark}")
    if not fam_rows:
        lines.append("  (no family has 2+ rows)")
    lines.append("")
    return lines


# Diversity / stratum tunables come from configs/config.yaml (one source).
CFG_DIVERSITY = CFG.get("diversity", {})
CFG_EVAL = CFG.get("evaluate", {})


def strata_report(df: pd.DataFrame) -> list[str]:
    """Counts + style mix + scam rate for every real_stratum (Phase 2.5).

    test_real is NEVER reported as one blended number - 31k translated IMC25
    scams would drown out your own messages. Each stratum is reported on its
    own, and anything under min_stratum_rows is marked too small for claims.
    """
    lines = ["--- real_strata (test_real broken into honest segments) ---"]
    min_rows = int(CFG_EVAL.get("min_stratum_rows", 30))
    real = df[df["split"] == "test_real"]
    if real.empty:
        lines.append("  test_real is EMPTY")
        lines.append("")
        return lines
    for strat in list(REAL_STRATA):
        sub = real[real["real_stratum"] == strat]
        n = len(sub)
        if n == 0:
            lines.append(f"  {strat:22s} 0 rows")
            continue
        scam = int((sub["verdict"] == "scam").sum())
        rate = scam / n
        styles = ", ".join(f"{k}={v}" for k, v in
                           sub["script_style"].value_counts().items())
        small = "  <-- TOO SMALL FOR CLAIMS" if n < min_rows else ""
        lines.append(f"  {strat:22s} {n:6d} rows | scam rate {rate:.1%} | "
                     f"styles: {styles}{small}")
    stray = int((~real["real_stratum"].isin(list(REAL_STRATA))).sum())
    if stray:
        lines.append(f"  WARNING: {stray} test_real row(s) with no real_stratum")
    lines.append("")
    return lines


def data_report(df: pd.DataFrame, out_path: Path | None = None) -> str:
    """Build the human-readable data report; save to txt and return it."""
    out_path = out_path or data_path("data_report")
    lines: list[str] = []
    add = lines.append

    add("=" * 70)
    add("ScamLens DATA REPORT (Phase 2.5)")
    add("=" * 70)
    add(f"rows: {len(df)}   columns: {len(df.columns)}")
    add("")

    add("--- verdict x source -------------------------------------------------")
    add(pd.crosstab(df["verdict"], df["source"]).to_string())
    add("")

    add("--- tactic frequency (multi-label, so counts can exceed row count) ---")
    tactic_counts: dict[str, int] = {}
    for val in df["tactics"]:
        for tag in as_tactics_list(val):
            tactic_counts[tag] = tactic_counts.get(tag, 0) + 1
    tc = pd.Series(tactic_counts).sort_values(ascending=False)
    add(tc.to_string() if len(tc) else "(no tactic tags yet)")
    untagged_tags = [t for t in TACTIC_TAGS if t not in tactic_counts]
    if untagged_tags:
        add(f"tactic tags with ZERO rows: {untagged_tags}")
    add("")

    add("--- script_style distribution ----------------------------------------")
    add(df["script_style"].value_counts().to_string())
    add("")

    add("--- url_type distribution --------------------------------------------")
    add(df["url_type"].value_counts().to_string())
    add("")

    add("--- message length stats (characters / words, scrubbed text) ---------")
    lens = pd.DataFrame({
        "chars": df["text"].astype(str).str.len(),
        "words": df["text"].astype(str).str.split().str.len(),
    })
    add(lens.describe().round(1).to_string())
    add("")

    add("--- cluster stats ----------------------------------------------------")
    sizes = df["template_cluster"].value_counts()
    add(f"clusters: {len(sizes)}")
    add(f"singleton clusters (size 1): {int((sizes == 1).sum())} "
        f"({(sizes == 1).mean():.0%} of clusters)")
    add(f"largest cluster size: {int(sizes.max()) if len(sizes) else 0}")
    hist = {int(k): int(v) for k, v in sizes.value_counts().sort_index().head(10).items()}
    add(f"size histogram (size -> #clusters): {hist}")
    add("")

    add("--- split balance ----------------------------------------------------")
    add(pd.crosstab(df["split"], df["verdict"]).reindex(SPLITS, fill_value=0).to_string())
    add(f"rows per split: {df['split'].value_counts().reindex(SPLITS, fill_value=0).to_dict()}")
    add("")

    if "real_stratum" in df.columns:
        add("--- real_strata ------------------------------------------------------")
        for line in strata_report(df)[:-1]:     # drop the trailing blank line
            add(line)
        add("")

    add("--- diversity: unique text ratio (Phase 2.5 duplicate control) -------")
    for line in diversity_report(df)[:-1]:      # drop the trailing blank line
        add(line)

    add("--- binary table check (suspicious = unsure band) --------------------")
    binary_problems = binary_table_check(df)
    if binary_problems:
        add("BINARY CHECK: FAIL")
        for p in binary_problems:
            add(f"  - {p}")
    else:
        add(f"BINARY CHECK: PASS (only {list(BINARY_VERDICTS)} present)")
    add("")

    add("--- leakage check ----------------------------------------------------")
    status, problems = leakage_check(df)
    add(f"LEAKAGE CHECK: {status}")
    for p in problems:
        add(f"  - {p}")
    add("")

    add("--- schema validation ------------------------------------------------")
    try:
        validate_schema(df)
        add("SCHEMA: PASS")
    except ValueError as e:
        add("SCHEMA: FAIL")
        add(str(e))

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n(report saved to {out_path})")
    return report


if __name__ == "__main__":
    print("src/data_checks.py")
    print("=" * 60)
    pq = data_path("messages_parquet")
    if not pq.exists():
        print(f"No {pq} yet. Run 'python -m src.build_dataset' first.")
    else:
        df = pd.read_parquet(pq)
        data_report(df)
        # Phase 2.5: these are not friendly warnings. A leakage or binary-table
        # failure makes every downstream number meaningless, so exit non-zero.
        status, problems = leakage_check(df)
        binary_problems = binary_table_check(df)
        # Synthetic uniqueness target (Phase 2.5 #1).
        min_ratio = float(CFG_DIVERSITY.get("min_unique_ratio", 0.95))
        norm = df["text"].map(normalize_text)
        syn = df[df["source"].astype(str).str.startswith("synthetic")]
        syn_ratio = (syn["text"].map(normalize_text).nunique() / len(syn)) if len(syn) else 1.0
        if syn_ratio < min_ratio:
            problems.append(
                f"synthetic unique-text ratio {syn_ratio:.1%} < {min_ratio:.0%} "
                "(expand src/synth/components.py and regenerate)")
        if status != "PASS" or binary_problems:
            print("\n" + "!" * 70)
            print("DATA CHECKS FAILED - DO NOT TRAIN OR REPORT METRICS")
            for p in problems + binary_problems:
                print("  -", p)
            print("!" * 70)
            raise SystemExit(1)
