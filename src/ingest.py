"""
src/ingest.py - private raw messages -> scrubbed unlabeled pool.

INPUT : data/private/raw_messages.txt   (gitignored, RAW, never copied anywhere)
OUTPUT: data/interim/unlabeled_pool.csv (scrubbed only, no verdict yet)

The raw file contains one message per "block", with blocks separated by a line
that contains only three dashes:
    ---
Lines starting with '#' are comments and ignored (handy for notes at the top
of the file).

For every message we:
  1. scrub out PII (phone/UPI/email/account/URL/name -> <PHONE> etc.)
  2. detect the script_style (Devanagari vs Latin, roman Hinglish vs English)
  3. remember whether the ORIGINAL text had a phone number (had_phone_number)
  4. write ONLY the scrubbed text to the interim CSV

The raw text never leaves data/private/. After running this you MUST manually
skim the CSV for leftover personal info - regex scrubbing is imperfect.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import SEED, data_path
from src.scrub import scrub
from src.config import HINGLISH_WORDS

# Columns of the unlabeled pool CSV. verdict/tactics stay EMPTY until you
# label them with src/label_cli.py.
POOL_COLUMNS = [
    "id", "text", "verdict", "tactics", "channel", "script_style",
    "date_received", "source", "labeler", "label_confidence",
    "url_type", "had_phone_number", "notes",
]

# A small set of "very Hinglish" words that alone are enough to call a message
# roman_hinglish even if it only contains one of them.
STRONG_HINGLISH = {"kripya", "kripaya", "turant", "karein", "jayega", "jaayega", "nahi", "nahin"}


def read_raw_messages(path: Path) -> list[str]:
    """Split the private raw file into individual messages on '---' lines."""
    text = path.read_text(encoding="utf-8")
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        # Comments and separators are not part of any message.
        if line.strip() == "---":
            blocks.append("\n".join(current).strip())
            current = []
        elif line.lstrip().startswith("#"):
            continue
        else:
            current.append(line)
    blocks.append("\n".join(current).strip())
    # Drop empty blocks (blank lines between separators).
    return [b for b in blocks if b]


def detect_script_style(text: str) -> str:
    """Guess how the message is written.

    We count Devanagari characters (the Hindi script, U+0900-U+097F) versus
    Latin letters. Then:
      - both scripts substantial        -> "mixed"
      - mostly Devanagari               -> "devanagari"
      - Latin + Hinglish function words -> "roman_hinglish"
      - Latin only                      -> "english"
    """
    devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = devanagari + latin

    if total == 0:
        return "english"  # nothing to go on; safe default

    if devanagari > 0 and latin > 0:
        ratio = devanagari / total
        # If neither script dominates, call it mixed.
        if 0.1 <= ratio <= 0.9:
            return "mixed"

    if devanagari > latin:
        return "devanagari"

    # Latin-dominant: check for Hinglish function words.
    words = {w.strip(".,!?;:()\"'").lower() for w in text.split()}
    hits = words & set(HINGLISH_WORDS)
    strong_hits = hits & STRONG_HINGLISH
    # >=2 distinct Hinglish words, or 1 very-Hinglish word, means Hinglish.
    if len(hits) >= 2 or strong_hits:
        return "roman_hinglish"
    return "english"


def ingest(raw_path: Path | None = None, out_path: Path | None = None) -> pd.DataFrame:
    """Run the full ingest step and return the pool DataFrame."""
    raw_path = raw_path or data_path("private_raw")
    out_path = out_path or data_path("unlabeled_pool")

    if not raw_path.exists():
        print(f"No raw file found at {raw_path}")
        print("-> Put your consented raw messages there (blocks separated by '---').")
        return pd.DataFrame(columns=POOL_COLUMNS)

    messages = read_raw_messages(raw_path)
    print(f"Found {len(messages)} message(s) in {raw_path}")

    rows = []
    n_leftover_warnings = 0
    for i, raw in enumerate(messages, start=1):
        clean, meta = scrub(raw)                       # step 1: remove PII
        style = detect_script_style(clean)             # step 2: writing system
        if meta["leftover_digits"]:
            n_leftover_warnings += 1
        rows.append({
            "id": f"p{i:04d}",                         # pool ids: p0001, p0002, ...
            "text": clean,                             # SCRUBBED text only
            "verdict": "",                             # filled later by label_cli
            "tactics": "",                             # filled later by label_cli
            "channel": "unknown",                      # you can set this in label_cli
            "script_style": style,
            "date_received": "",                       # optional, set during labeling
            "source": "self_collected",                # default; change to family_friends if needed
            "labeler": "",
            "label_confidence": "",
            "url_type": meta["url_type"],
            "had_phone_number": bool(meta["had_phone_number"]),
            "notes": "",
        })

    df = pd.DataFrame(rows, columns=POOL_COLUMNS)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Ingested {len(df)} message(s) -> {out_path}")
    print(f"  script_style counts: {df['script_style'].value_counts().to_dict()}")
    print(f"  messages with a phone-number pattern: {int(df['had_phone_number'].sum())}")
    if n_leftover_warnings:
        print(f"  WARNING: {n_leftover_warnings} message(s) contain 5+ digit runs the scrubber "
              f"could not classify - open the CSV and check them manually.")
    print()
    print("PRIVACY REMINDER: please manually skim", out_path)
    print("for any leftover personal information BEFORE labeling. The scrubber is imperfect.")
    return df


if __name__ == "__main__":
    print(f"(seed={SEED}) src/ingest.py")
    print("=" * 60)
    ingest()
