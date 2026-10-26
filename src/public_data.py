"""
src/public_data.py - optional weak PUBLIC data (fully verified sources only).

WHY "WEAK"
----------
The SMS Spam Collection (UCI) is English commercial spam (Viagra, cheap loans,
free prizes). Indian scam SMS is a different animal. So:
    ham  -> safe          (legit English SMS)
    spam -> suspicious    (NOT scam - commercial spam is not clear fraud)
    source = "public_weak", label_confidence = "low"
These rows may NEVER appear in the val/test splits or in any golden set
(src/splits.py hard-forces them into train).

DATASETS KEPT / SKIPPED  (verified on 2026-10-20, see README for sources)
-----------------------------------------------------------------------
KEPT:
  1. UCI SMS Spam Collection via HuggingFace id "ucirvine/sms_spam".
     License CC BY 4.0 (per UCI; the HF card says "unknown"). 5,574 rows,
     columns sms / label. English, not Hinglish.

SKIPPED (each failed at least one verification bar):
  2. "Indian Telecom SMS Spam Collection" (github.com/junioralive/
     india-spam-sms-classification, MIT). EXISTS and license is clear, but
     the text is mostly Indian-English commercial SMS - not Hinglish scam
     messages. Listed here so you can revisit it later.
  3. "SMS Phishing Dataset" (Mendeley Data, f45bkkt8pr). No license stated on
     the record; not established as Indian or Hinglish. Skipped.
  4. "Smishing-Dataset-IMC25" (github.com/reportsmishing, CC BY 4.0). Real
     smishing reports, but global not Indian, language mix not Hinglish.
     Revisit later if you want English smishing negatives.
  5. "SMSDHL" (TechRxiv). Paper not accessible (403); no dataset file or
     license verifiable. Skipped - we do not guess dataset ids.
  6. Kaggle "SMS SPAM DATASET" (tinu10kumar). No license stated; contents
     unverified. Skipped.

BOTTOM LINE: no verifiable PUBLIC dataset of Indian/Hinglish scam SMS exists
that we could find. YOUR consensually collected messages are the real corpus.

Run:  python -m src.public_data          (tries to download; skips politely)
      python -m src.public_data --limit 500   (smaller sample)
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import CFG, SEED, data_path
from src.ingest import detect_script_style
from src.scrub import scrub

PUBLIC_COLUMNS = [
    "id", "text", "verdict", "tactics", "channel", "script_style",
    "date_received", "source", "labeler", "label_confidence",
    "url_type", "had_phone_number", "notes",
]

# The research findings, printed every run so the decision is transparent.
CANDIDATES = [
    {
        "name": "UCI SMS Spam Collection (ucirvine/sms_spam)",
        "license": "CC BY 4.0 (per UCI; HF card says unknown)",
        "hinglish": "NO - English",
        "verdict": "KEEP (weak)",
        "why": "Verified to exist; English commercial spam only, so spam->suspicious.",
    },
    {
        "name": "Indian Telecom SMS Spam Collection (github.com/junioralive/india-spam-sms-classification)",
        "license": "MIT",
        "hinglish": "NO - mostly Indian-English commercial SMS",
        "verdict": "SKIP",
        "why": "India-relevant but not Hinglish and not scam; revisit later if wanted.",
    },
    {
        "name": "SMS Phishing Dataset (Mendeley Data f45bkkt8pr)",
        "license": "not stated on the record",
        "hinglish": "not verified",
        "verdict": "SKIP",
        "why": "No license stated; not established as Indian/Hinglish.",
    },
    {
        "name": "Smishing-Dataset-IMC25 (github.com/reportsmishing)",
        "license": "CC BY 4.0",
        "hinglish": "not verified (global reports, not India-specific)",
        "verdict": "SKIP",
        "why": "Real smishing but not Indian/Hinglish; revisit for English negatives.",
    },
    {
        "name": "SMSDHL (TechRxiv paper)",
        "license": "could not verify (paper fetch 403)",
        "hinglish": "not verified",
        "verdict": "SKIP",
        "why": "No accessible dataset file; we never guess dataset ids.",
    },
    {
        "name": "Kaggle 'SMS SPAM DATASET' (tinu10kumar)",
        "license": "not stated",
        "hinglish": "unknown",
        "verdict": "SKIP",
        "why": "No license stated and contents unverified.",
    },
]


def print_candidates() -> None:
    """Print the verified-candidates table so you always know what is in/out."""
    print("PUBLIC DATASET RESEARCH (each verified by hand):")
    for c in CANDIDATES:
        print(f"  [{c['verdict']:<11}] {c['name']}")
        print(f"       license : {c['license']}")
        print(f"       hinglish: {c['hinglish']}")
        print(f"       why     : {c['why']}")
    print()


def load_sms_spam(limit: int | None = None) -> pd.DataFrame:
    """Try to download the UCI SMS Spam Collection from HuggingFace.

    Returns an empty DataFrame if the download fails (no internet, HF down,
    dataset renamed...). We NEVER guess alternative ids - we just skip.
    """
    dataset_id = CFG["public_data"]["sms_spam_dataset_id"]
    try:
        from datasets import load_dataset  # imported here so a missing lib only affects this step

        print(f"Trying to load '{dataset_id}' from HuggingFace...")
        try:
            ds = load_dataset(dataset_id, split="train")
        except Exception:
            # Older copies of this dataset used a config name; try that once.
            ds = load_dataset(dataset_id, "plain_text", split="train")
        df = pd.DataFrame({"text": ds["sms"], "label": ds["label"]})
        print(f"  downloaded {len(df)} rows. label counts: {df['label'].value_counts().to_dict()}")
        if limit:
            # Deterministic sample (seed 42) so runs are reproducible.
            df = df.sample(n=min(limit, len(df)), random_state=SEED).reset_index(drop=True)
            print(f"  limited to {len(df)} rows (deterministic sample, seed={SEED}).")
        return df
    except Exception as e:  # noqa: BLE001 - we want to catch literally any download error
        print(f"  SKIPPED: could not load '{dataset_id}': {type(e).__name__}: {e}")
        print("  (That is fine - this step is optional.)")
        return pd.DataFrame(columns=["text", "label"])


def to_weak_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw ham/spam rows into our schema, with weak label mapping.

        ham  -> safe
        spam -> suspicious   (commercial spam is NOT proven fraud -> not 'scam')

    Every row: source=public_weak, confidence=low, tactic tags EMPTY (we did
    not read them all, so we do not invent tactic labels).
    """
    def is_spam(label) -> bool:
        # The HF copy returns labels as integers (1=spam, 0=ham); older copies
        # used the strings 'spam'/'ham'. Accept both so we never mislabel.
        return str(label).strip().lower() in {"1", "spam"}

    rows = []
    for i, r in df.iterrows():
        clean, meta = scrub(str(r["text"]))   # even public text gets scrubbed
        verdict = "suspicious" if is_spam(r["label"]) else "safe"
        rows.append({
            "id": f"pub{i + 1:05d}",
            "text": clean,
            "verdict": verdict,
            "tactics": "",                       # never invent tactic tags
            "channel": "sms",
            "script_style": detect_script_style(clean),
            "date_received": "",
            "source": "public_weak",
            "labeler": "public_rule_ham_spam",
            "label_confidence": "low",
            "url_type": meta["url_type"],
            "had_phone_number": bool(meta["had_phone_number"]),
            "notes": "UCI SMS Spam Collection; English; NEVER use in val/test/golden",
        })
    out = pd.DataFrame(rows, columns=PUBLIC_COLUMNS)
    print(f"Converted {len(out)} public rows -> weak labels "
          f"{out['verdict'].value_counts().to_dict()}")
    return out


def main() -> None:
    print(f"(seed={SEED}) src/public_data.py")
    print("=" * 60)
    print_candidates()

    limit = None
    if "--limit" in sys.argv:
        pos = sys.argv.index("--limit")
        limit = int(sys.argv[pos + 1])

    raw = load_sms_spam(limit=limit)
    out_path = data_path("public_weak")
    if raw.empty:
        print("No public data downloaded; not writing", out_path)
        return
    weak = to_weak_rows(raw)
    weak.to_csv(out_path, index=False, encoding="utf-8")
    print("Wrote", out_path)
    print("REMINDER: source=public_weak rows are forced into the TRAIN split only")
    print("and must never be used for evaluation or the golden set.")


if __name__ == "__main__":
    main()
