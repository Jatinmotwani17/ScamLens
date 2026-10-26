"""
src/config.py - the single place where constants, enums and paths live.

Why: every later phase (labeling, training, API) must agree on the exact
verdict names, tactic tag names, and column names. If they are defined once
here, a typo in one file cannot silently create a second vocabulary.

This file also loads configs/config.yaml (tunables like seed and thresholds).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the scamlens/ folder itself (one level above src/).
# All file paths in the project are built from it, so scripts work no matter
# which folder you run them from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read configs/config.yaml and return it as a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# The one config dictionary used by every module.
CFG = load_config()

# Reproducibility seed. Used by: dedup (MinHash), splits (shuffling), synthetic.
SEED = int(CFG.get("seed", 42))


def data_path(key: str) -> Path:
    """Return an absolute Path for a named file from config.yaml's `paths:` block.

    Folders are created automatically so beginners never hit 'folder not found'.
    """
    rel = CFG["paths"][key]
    p = PROJECT_ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# TAXONOMY - the label vocabulary (see docs/TAXONOMY.md for full definitions)
# ---------------------------------------------------------------------------

# The three top-level verdicts. A message gets exactly one.
VERDICTS = ("scam", "suspicious", "safe")

# Phase 2.5: "suspicious" is an UNSURE BAND, not a class we train or
# evaluate. Models and metrics are BINARY (scam vs safe); rows labelled
# suspicious are moved to data/processed/suspicious_holdout.parquet and come
# back in Phase 4 as a confidence-based abstention test.
BINARY_VERDICTS = ("safe", "scam")

# Strata (segments) of test_real, so we never report ONE blended number for
# "real data" - 31k English IMC25 scams would drown out your own messages.
#   user_real_scam  - YOUR messages labeled scam
#   user_real_safe  - YOUR messages labeled safe
#   external_scam   - IMC25 user-reported smishing (translated English)
#   public_ham_english - real legit messages from the UCI ham reserve
# Non-real rows have real_stratum = "" (empty string).
REAL_STRATA = ("user_real_scam", "user_real_safe", "external_scam",
               "public_ham_english")

# Multi-label tactic tags: a scam message can carry several of these.
TACTIC_TAGS = (
    "otp_or_pin_request",
    "urgency_threat",
    "bank_kyc_impersonation",
    "authority_impersonation",      # police/CBI/customs/telecom, incl. "digital arrest"
    "payment_or_upi_request",       # collect requests, "send Rs 1 to verify"
    "suspicious_link",
    "prize_or_lottery",
    "job_or_task_scam",
    "investment_or_trading_scam",
    "loan_app_harassment",
    "parcel_courier_scam",
    "utility_disconnection_scam",   # electricity / gas / SIM disconnection
    "relative_in_trouble",          # "Papa new number", fake child-in-trouble
)

# Where the message came from. Only 'synthetic' / 'synthetic_llm' rows are
# machine-generated. HARD RULE: synthetic rows are NEVER labelled as real.
#   external_reported - REAL user-reported scams from public datasets (IMC25)
#   public_ham_eval   - REAL legit messages reserved for test_real (UCI ham)
SOURCES = ("self_collected", "family_friends", "public_weak", "synthetic",
           "synthetic_llm", "external_reported", "public_ham_eval")

# What kind of channel the message arrived on.
CHANNELS = ("sms", "whatsapp", "email", "call_transcript", "unknown")

# Writing system / language style of the text.
SCRIPT_STYLES = ("roman_hinglish", "devanagari", "english", "mixed")

# How sure the labeler is. Used to down-weight or drop rows later.
CONFIDENCE_LEVELS = ("high", "medium", "low")

# What kind of URL the message contained (a strong scam signal, see scrub.py).
URL_TYPES = ("none", "shortener", "ip_address", "odd_tld", "official_looking", "other")

# Which split a row belongs to (Phase 2 five-way split):
#   train/val/test_a - model development (synthetic allowed)
#   test_b           - unseen template FAMILIES (new-variant test)
#   test_real        - REAL human-labeled messages only; headline numbers
#                      always come from here, never from synthetic splits.
# HARD RULE: a synthetic row may never sit in test_real.
SPLITS = ("train", "val", "test_a", "test_b", "test_real")

# ---------------------------------------------------------------------------
# DATA SCHEMA - the contract for data/processed/messages.parquet
# ---------------------------------------------------------------------------
# Column order matters: this is exactly the order written to parquet.

MESSAGE_COLUMNS = [
    "id",               # unique string id, e.g. "p0001" (pool), "syn0001", "pub0001"
    "text",             # SCRUBBED text only - no raw phone numbers/emails/URLs
    "verdict",          # one of VERDICTS
    "tactics",         # list of tag strings from TACTIC_TAGS (may be empty)
    "channel",          # one of CHANNELS
    "script_style",     # one of SCRIPT_STYLES
    "date_received",    # "YYYY-MM-DD" or "YYYY-MM" or None if unknown
    "source",           # one of SOURCES
    "template_family",  # string: template family name (e.g. "kyc_f1"); "" = no family
    "labeler",          # free-text name of who/what assigned the label
    "label_confidence", # one of CONFIDENCE_LEVELS
    "url_type",         # one of URL_TYPES
    "had_phone_number", # bool: did the ORIGINAL text contain a phone number
    "template_cluster", # int: cluster id from near-duplicate detection
    "split",            # one of SPLITS
    "real_stratum",     # one of REAL_STRATA, or "" when the row is not real
    "notes",            # free-text notes (empty string if none)
]

# Pandas dtypes we expect for the simple (non-list) columns.
# Used by validate_schema to give clearer error messages.
COLUMN_DTYPE_HINTS = {
    "had_phone_number": "bool",
    "template_cluster": "int",
}

# ---------------------------------------------------------------------------
# LANGUAGE HELPERS
# ---------------------------------------------------------------------------
# A small hand-picked list of common Hindi function words written in Latin
# script ("roman Hinglish"). If a Latin-script message contains these words,
# it is almost certainly Hinglish rather than plain English.
HINGLISH_WORDS = (
    "aap", "aapka", "aapke", "hai", "hain", "nahi", "nahin", "karein",
    "kripya", "kripaya", "turant", "jayega", "jaayega", "karo", "karna",
    "matlab", "bhai", "paisa", "rupaye", "rupay", "rs", "kar", "raha",
    "rahe", "hoon", "hun", "kya", "acha", "accha", "theek", "thik",
    "bhej", "bhejo", "lena", "dena", "madad", "bacha", "beta", "papa",
    "mai", "mera", "meri", "tum", "tera", "apna", "jaldi", "abhi",
)

# Words that look like the regex "Dear X" pattern but are NOT person names.
NAME_STOPWORDS = {
    "OK", "I", "A", "An", "The", "Team", "Bank", "Care", "Support",
    "Hello", "Hi", "User", "Customer", "Member", "Dear", "Sir", "Madam",
    "OTP", "SMS", "KYC", "UPI", "ATM", "PIN", "Dept", "Department",
    # Phase 2: words that show up in bank-style greetings/sign-offs but are
    # NOT person names ("Dear Account Holder", "Regards, ... Holder Team").
    "Account", "Holder", "Winner", "Grahak", "Client", "Applicant",
}

# TLDs that look normal/official; everything else is treated as "odd".
OFFICIAL_TLDS = {
    "com", "co", "in", "co.in", "org", "net", "gov.in", "nic.in",
    "ac.in", "edu", "edu.in", "mil", "int", "io", "me", "tv",
}

# TLDs commonly seen on cheap throwaway scam landing pages.
ODD_TLDS = {
    "xyz", "top", "click", "tk", "cf", "gq", "ml", "zip", "mov",
    "work", "loan", "win", "buzz", "rest", "icu", "cam", "surf",
}

# Well-known URL shorteners - the destination is hidden, so treat as signal.
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rb.gy", "shorturl.at", "v.gd", "s.id",
    "bl.ink", "trib.al", "ift.tt",
}

# Brand / authority tokens. If one appears inside a URL hostname, the link is
# "official-looking" (e.g. a fake bank page). Rough heuristic on purpose.
BRAND_TOKENS = (
    "sbi", "hdfc", "icici", "axis", "kotak", "idbi", "pnb", "bob",
    "unionbank", "canarabank", "paytm", "phonepe", "gpay", "bhim",
    "airtel", "jio", "vi-", "vodafone", "bsnl", "irctc", "india-post",
    "incometax", "gst", "rbi", "trai", "cyber", "passport", "aadhaar",
    "uidai", "lic", "npci", "digi", "postpe", "lazypay", "kredit",
)

if __name__ == "__main__":
    # Tiny self-check you can run: python -m src.config
    print("ScamLens config loaded.")
    print("  project root :", PROJECT_ROOT)
    print("  seed         :", SEED)
    print("  verdicts     :", VERDICTS)
    print("  tactic tags  :", len(TACTIC_TAGS))
    print("  schema cols  :", len(MESSAGE_COLUMNS))
