"""
src/synth/llm_paraphrase.py - OPTIONAL naturalness pass using Gemini.

WHY: template composition produces slightly robotic Hinglish. This script
rewrites a random 400-message subset into more natural Hinglish so Phase 3
models see varied phrasing. It is completely OPTIONAL:

  * no GEMINI_API_KEY in .env  -> prints "Skipping LLM paraphrasing" and
    exits cleanly. The pipeline works fine without it.
  * every response is cached on disk (data/llm_cache/), so re-running costs
    nothing and makes zero new API calls.
  * a hard cap (llm.max_calls in config.yaml) is printed BEFORE running.

HONESTY RULES:
  * outputs keep source="synthetic_llm" (never "synthetic", NEVER real).
    They inherit verdict / tactics / template_family from the original row.
  * the paraphrase MUST keep <URL> / <PHONE> / <UPI> placeholders exactly
    and must not invent new links/numbers. Rows that fail the mechanical
    checks (placeholder drift, length blow-up, refusal text, new digits)
    are DROPPED and counted loudly - we do not silently keep dodgy rows.
  * this is still synthetic data. It may sit in train/val/test_a/test_b but
    NEVER in test_real.

Commands:
  python -m src.synth.llm_paraphrase            # paraphrase the subset
  python -m src.synth.llm_paraphrase --review   # print 30 pairs for YOUR eyes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG, SEED, data_path

LLM_CFG = CFG["llm"]

# The safety placeholders that must survive paraphrasing VERBATIM.
PLACEHOLDER_TOKENS = ("<URL>", "<PHONE>", "<UPI>")

# Phrases that mean the model refused instead of paraphrasing.
REFUSAL_STARTS = ("i can't", "i cannot", "i'm sorry", "sorry,", "as an ai",
                  "i am sorry", "unfortunately")

NEW_DIGIT_RUN = re.compile(r"\d{6,}")     # new long digit runs are suspicious
URL_LIKE = re.compile(r"https?://|www\.", re.IGNORECASE)

PROMPT = """You are helping build training data for a scam-detection classifier \
for Indian SMS/WhatsApp messages (English and Hinglish).

Rewrite the message below so it sounds like a natural message a real person \
would type or a scammer would send: conversational Hinglish is preferred, \
with realistic spelling, but keep the MEANNING and the INTENT identical.

STRICT RULES:
1. Keep the placeholders <URL>, <PHONE>, <UPI> EXACTLY as they appear. Never replace them.
2. Never add a real website address, phone number, UPI id or person name.
3. Do not add or remove the ask (OTP request, link click, payment, call...).
4. Keep it the same kind of message (a scam stays a scam, a bank alert stays a bank alert).
5. Output ONLY the rewritten message text. No explanations, no quotes.

MESSAGE:
{message}
"""


# ---------------------------------------------------------------------------
# API key + cache
# ---------------------------------------------------------------------------
def have_api_key() -> str | None:
    """Return the Gemini API key from .env / environment, or None."""
    try:
        from dotenv import load_dotenv
        load_dotenv()                                    # reads ./.env
    except Exception:
        pass                                             # dotenv optional
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def cache_path(text: str) -> Path:
    """One cache file per source text (content hash), so reruns are free."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    folder = data_path("llm_cache_dir")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.json"


def read_cache(text: str) -> str | None:
    p = cache_path(text)
    if p.exists():
        try:
            return str(json.loads(p.read_text(encoding="utf-8"))["paraphrase"])
        except Exception:
            return None
    return None


def write_cache(text: str, paraphrase: str) -> None:
    cache_path(text).write_text(
        json.dumps({"source": text, "paraphrase": paraphrase}, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Quality gate (mechanical, honest, documented)
# ---------------------------------------------------------------------------
def check_paraphrase(original: str, new: str) -> str | None:
    """Return None if OK, else the reason to drop the row."""
    if not new or not new.strip():
        return "empty output"
    low = new.strip().lower()
    if any(low.startswith(s) for s in REFUSAL_STARTS):
        return "model refusal"
    for tok in PLACEHOLDER_TOKENS:
        if tok in original and tok not in new:
            return f"lost placeholder {tok}"
    if URL_LIKE.search(new):
        return "invented a URL"
    if NEW_DIGIT_RUN.search(new):
        return "invented a long digit run"
    ratio = len(new) / max(len(original), 1)
    if ratio < 0.33 or ratio > 3.0:
        return f"length changed too much (x{ratio:.1f})"
    return None


# ---------------------------------------------------------------------------
# One API call (wrapped so missing package = clean skip)
# ---------------------------------------------------------------------------
def make_client(api_key: str):
    """Create a google-genai client, or explain exactly what is missing."""
    try:
        from google import genai            # type: ignore
    except Exception:
        print("The 'google-genai' package is not installed.")
        print("  Fix:  pip install google-genai python-dotenv")
        return None
    return genai.Client(api_key=api_key)


def paraphrase_one(client, text: str) -> str | None:
    """Paraphrase ONE message. Returns None on any API failure."""
    cached = read_cache(text)
    if cached is not None:
        return cached
    try:
        resp = client.models.generate_content(
            model=LLM_CFG["model"], contents=PROMPT.format(message=text)
        )
        out = (resp.text or "").strip()
    except Exception as e:                      # network / quota / safety
        print(f"    API error: {type(e).__name__}: {e}")
        return None
    if out:
        write_cache(text, out)                  # cache EVERY response
    return out or None


# ---------------------------------------------------------------------------
# Review mode - 30 random pairs for a human to eyeball
# ---------------------------------------------------------------------------
def review(n: int = 30) -> None:
    """Print n random (original, paraphrase) pairs from synthetic_llm.parquet."""
    llm_path = data_path("synthetic_llm_parquet")
    if not llm_path.exists():
        print(f"No paraphrases found at {llm_path}.")
        print("Run  python -m src.synth.llm_paraphrase  first (needs GEMINI_API_KEY).")
        return
    df = pd.read_parquet(llm_path)
    if len(df) == 0:
        print("synthetic_llm.parquet is empty - nothing to review.")
        return
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    print("=" * 70)
    print(f"REVIEW {len(idx)} RANDOM PARAPHRASES - say if any changed meaning")
    print("=" * 70)
    for i in sorted(idx):
        row = df.iloc[i]
        print(f"\n--- {row['id']}  [{row['verdict']}]  family={row['template_family']}")
        print(f"  ORIG: {row['notes_original_text']}")
        print(f"  PARA: {row['text']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Optional Gemini paraphrase pass.")
    parser.add_argument("--review", action="store_true",
                        help="print 30 random paraphrases for manual review")
    parser.add_argument("--subset-size", type=int, default=int(LLM_CFG["subset_size"]))
    args = parser.parse_args(argv)

    if args.review:
        review()
        return

    api_key = have_api_key()
    if api_key is None:
        print("Skipping LLM paraphrasing (no GEMINI_API_KEY in .env).")
        print("  The rest of the pipeline continues without it.")
        return
    client = make_client(api_key)
    if client is None:
        return

    # Load the synthetic corpus (already generated by src/synth/generate.py).
    syn_path = data_path("synthetic_parquet")
    if not syn_path.exists():
        raise SystemExit(f"{syn_path} not found. Run: python -m src.synth.generate")
    df = pd.read_parquet(syn_path)

    # Random subset, seeded - and how many calls a cold run would make.
    rng = np.random.default_rng(SEED)
    subset_n = min(args.subset_size, len(df))
    idx = rng.choice(len(df), size=subset_n, replace=False)
    subset = df.iloc[sorted(idx)].reset_index(drop=True)

    calls_needed = sum(1 for t in subset["text"] if read_cache(str(t)) is None)
    cap = int(LLM_CFG["max_calls"])
    print(f"(seed={SEED}) src/synth/llm_paraphrase.py")
    print("=" * 60)
    print(f"subset size      : {subset_n} messages")
    print(f"already cached   : {subset_n - calls_needed}")
    print(f"API calls needed : {min(calls_needed, cap)} "
          f"(HARD CAP = {cap}, cache makes reruns free)")
    if calls_needed == 0:
        print("Nothing to do - every response is cached.")
    elif calls_needed > cap:
        print(f"WARNING: {calls_needed} needed > cap {cap}; the rest stay unparaphrased.")

    rows, dropped, errors, done = [], 0, 0, 0
    for _, row in subset.iterrows():
        if done >= cap:
            break
        text = str(row["text"])
        out = paraphrase_one(client, text)
        if out is None:
            errors += 1
            continue
        done += 1
        reason = check_paraphrase(text, out)
        if reason is not None:
            dropped += 1
            print(f"  DROPPED {row['id']}: {reason}")
            continue
        new_row = row.to_dict()
        new_row["id"] = f"{row['id']}p"
        new_row["text"] = out
        new_row["source"] = "synthetic_llm"        # HARD RULE: never 'real'
        new_row["labeler"] = LLM_CFG["model"]
        new_row["label_confidence"] = "medium"      # meaning re-checked by YOU
        new_row["notes_original_text"] = text       # kept for --review
        rows.append(new_row)

    if rows:
        out_df = pd.DataFrame(rows)
        out_path = data_path("synthetic_llm_parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"\nWROTE {out_path}")
        print(f"  paraphrases kept : {len(out_df)}")
        print(f"  dropped (gate)   : {dropped}")
        print(f"  API errors       : {errors}")
        print("\nNext: eyeball them!  python -m src.synth.llm_paraphrase --review")
    else:
        print("\nNo paraphrases produced (all failed or capped). Nothing written.")


if __name__ == "__main__":
    main()
