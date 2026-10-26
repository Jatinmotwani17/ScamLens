"""
src/robustness.py - Phase 4 robustness PROBES (perturbation / transfer / ham).

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
  Three probes, all deliberately cheap and all deliberately humble:
    a. PERTURBATION  - re-score held-out scams after mechanical edits a scammer
                       makes for free (typos, spaced-out words, lookalikes,
                       leetspeak, ALL CAPS, emoji, hxxp/[.] links, extra
                       punctuation). Does recall survive the edit?
    b. HINGLISH TRANSFER - rewrite held-out REAL English scams into roman
                       Hinglish with a phrase dictionary built from the
                       roman_hinglish pools in src/synth/components.py. This is
                       a translation-derived probe, NOT real Hinglish traffic.
    c. SAFE-CHAT     - 200 fresh legitimate messages (family chat, delivery
                       coordination, bank-like alerts) from template components
                       that appear in NO training family. What share does the
                       model call scam?

WHY EVERY NUMBER HERE IS LABELLED "probe"
  An attack probe measures the model on text it has never seen, produced by a
  script we wrote. It says nothing about what real scammers do next quarter,
  and nothing about production traffic. So every table title, every row and
  the parquet itself say `robustness probe, not a real-world claim`.

NEVER TRAINED ON
  Every probe row gets source="probe_<kind>" and is written ONLY to
  data/processed/probes.parquet. Nothing in this file touches
  messages.parquet, so a probe row cannot reach a training split. The test
  suite asserts both facts.

MODEL-AGNOSTIC
  --model reads outputs/<name>/preds_<split>.parquet for a transformer run, or
  scores text on the fly for a TF-IDF baseline (verdict_lr_plus by default).
  Decisions use the FIXED operating point from configs/operating_point.yaml,
  which was fitted on validation only by src/calibrate.py.

Run:
  python -m src.robustness --model verdict_lr_plus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibrate import load_operating_point
from src.config import CFG, SEED, data_path
from src.io_utils import load_messages
from src.nn_sim import max_similarity_to_train
from src.phase4_common import (
    MIN_ROWS_FOR_CLAIMS,
    apply_temperature,
    claim_flag,
    load_predictor,
    too_small,
)

PROBE_PREFIX = "probe_"
P4 = CFG.get("phase4", {})
N_TRANSFER = int(P4.get("probe_n_transfer", 300))
N_SAFE_CHAT = int(P4.get("probe_n_safe_chat", 200))
REVIEW_N = int(P4.get("probe_review_n", 30))
# Held-out real scams sampled for the attack sweep (scoring 3.5k x 8 would be
# slow for no extra information).
N_ATTACK_REAL = 500

# ---------------------------------------------------------------------------
# (a) PERTURBATION ATTACKS
#
# Each attack edits a realistic SHARE of the message, not every token: an
# attacker who mangles 100% of words is not evading a human either. The rate
# is documented per attack so the number can be reproduced.
# ---------------------------------------------------------------------------
LOOKALIKES = {"a": "@", "e": "3", "i": "1", "o": "0", "c": "(", "s": "$"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "b": "8"}
EMOJIS = ["\U0001F525", "\U0001F4B0", "\U0001F600", "\u2757", "\U0001F4E6",
          "\u2B50", "\U0001F449", "\U0001F680"]
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|in|info|xyz|top|co|ly)\b)",
    re.IGNORECASE)
PLACEHOLDER_URL = "<URL>"
# Used only when the row's link was already scrubbed to <URL>: the probe then
# shows the same obfuscation a scammer types instead of a clickable link.
OBFUSCATED_URL = "hxxps://kyc-verify[.]info/claim"
WORD_RE = re.compile(r"[A-Za-z]+")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _map_words(text: str, fn, rate: float, rng) -> str:
    """Apply `fn` to a random `rate` share of the alphabetic words.

    Placeholder tokens (<URL>, <PHONE>, <DATE_TIME> ...) are NEVER edited:
    they are the project's guarantee that no real number or link leaks in, and
    an attack that mangles them would be testing something else.
    """
    spans = [(m.start(), m.end()) for m in re.finditer(r"<[^>]*>", text)]

    def inside_placeholder(start: int) -> bool:
        return any(lo <= start < hi for lo, hi in spans)

    out: list[str] = []
    last = 0
    for match in WORD_RE.finditer(text):
        out.append(text[last:match.start()])
        word = match.group(0)
        if (len(word) >= 3 and not inside_placeholder(match.start())
                and rng.random() < rate):
            out.append(fn(word, rng))
        else:
            out.append(word)
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def _swap(word: str, rng) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def attack_typos(text: str, rng) -> str:
    """Swap two adjacent characters in ~20% of words (phone-typing error)."""
    return _map_words(text, _swap, 0.20, rng)


def _space_inside(word: str, rng) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    return word[:i] + " " + word[i:]


def attack_spaces_inside_words(text: str, rng) -> str:
    """'account' -> 'acc ount' in ~20% of words (classic token-split evasion)."""
    return _map_words(text, _space_inside, 0.20, rng)


def _lookalike(word: str, rng) -> str:
    positions = [i for i, ch in enumerate(word) if ch.lower() in LOOKALIKES]
    if not positions:
        return word
    i = int(rng.choice(positions))
    return word[:i] + LOOKALIKES[word[i].lower()] + word[i + 1:]


def attack_lookalikes(text: str, rng) -> str:
    """One homoglyph per edited word (o->0, a->@, e->3 ...) in 30% of words."""
    return _map_words(text, _lookalike, 0.30, rng)


def _leet(word: str, rng) -> str:
    return "".join(LEET.get(ch.lower(), ch) for ch in word)


def attack_leetspeak(text: str, rng) -> str:
    """Full leetspeak on ~25% of words (a->4, e->3, o->0, s->5, t->7)."""
    return _map_words(text, _leet, 0.25, rng)


def attack_all_caps(text: str, rng) -> str:
    """Whole message in capitals, the cheapest urgency signal there is."""
    return text.upper()


def attack_emoji(text: str, rng) -> str:
    """Insert 2-4 emoji at word boundaries."""
    words = text.split(" ")
    for _ in range(int(rng.integers(2, 5))):
        pos = int(rng.integers(0, len(words) + 1))
        words.insert(pos, str(rng.choice(EMOJIS)))
    return " ".join(words)


def attack_url_obfuscation(text: str, rng) -> str:
    """Rewrite the link as hxxps://site[.]tld, the standard evasion."""
    def obf(match: re.Match) -> str:
        url = match.group(0)
        url = re.sub(r"^https?://", "hxxps://", url, flags=re.IGNORECASE)
        url = re.sub(r"^http://", "hxxp://", url, flags=re.IGNORECASE)
        return url.replace(".", "[.]")

    new, n = URL_RE.subn(obf, text)
    if n == 0 and PLACEHOLDER_URL in text:
        # The row's URL was scrubbed before it reached us; show the same
        # obfuscation a scammer would type instead.
        new = text.replace(PLACEHOLDER_URL, OBFUSCATED_URL, 1)
        n = 1
    return new


PUNCT = ["!", "!!", "...", "??", ",,,", " -", "*"]


def attack_punctuation(text: str, rng) -> str:
    """Sprinkle extra punctuation (!!, ..., ???) through the message."""
    words = text.split(" ")
    for _ in range(int(rng.integers(3, 7))):
        pos = int(rng.integers(0, len(words)))
        words[pos] = words[pos] + str(rng.choice(PUNCT))
    return " ".join(words)


ATTACKS: list[tuple[str, str, object]] = [
    ("typos", "random typos (char swap, 20% of words)", attack_typos),
    ("spaces_inside_words", "space inside words (20%)", attack_spaces_inside_words),
    ("lookalikes", "lookalike characters (30%)", attack_lookalikes),
    ("leetspeak", "leetspeak (25%)", attack_leetspeak),
    ("all_caps", "ALL CAPS", attack_all_caps),
    ("emoji", "emoji insertion", attack_emoji),
    ("url_obfuscation", "URL obfuscation (hxxp, [.])", attack_url_obfuscation),
    ("punctuation", "punctuation insertion", attack_punctuation),
]


# ---------------------------------------------------------------------------
# (b) HINGLISH TRANSFER - phrase dictionary from the synthetic pools
# ---------------------------------------------------------------------------
def _component_pools() -> list[dict]:
    """Every component list in src/synth/components.py that has en+roman text."""
    from src.synth import components as C

    pools: list[dict] = []
    for name in ("GREETINGS", "CONTEXTS", "TAILS", "CLOSINGS",
                 "SAFE_CONTEXTS", "SAFE_TAILS", "EASY_CHATTER"):
        pools.extend(getattr(C, name, []))
    for family in C.all_families():
        pools.extend(family["hooks"])
        pools.extend(family["demands"])
    return [c for c in pools
            if c["text"].get("english") and c["text"].get("roman_hinglish")]


def _strip_punct(text: str) -> str:
    return text.strip().strip(".,!?;:-").strip()


def phrase_dictionary() -> list[tuple[str, str]]:
    """(english phrase, roman-Hinglish phrase) pairs, longest first.

    Built entirely from components.py: every pair is one component's English
    text next to the SAME component's roman_hinglish text.
    """
    pairs: dict[str, str] = {}
    for comp in _component_pools():
        en = _strip_punct(comp["text"]["english"])
        hi = _strip_punct(comp["text"]["roman_hinglish"])
        if not en or not hi or en.lower() == hi.lower():
            continue
        pairs.setdefault(en.lower(), hi)
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


# A hand-aligned seed of scam/safe formula language. It exists because the
# pool-derived phrases are whole sentences and cover little of a REAL message;
# the seed covers the words that actually repeat. Every roman form below is
# checked against the pools at build time and silently dropped if it is not
# there, so no Hinglish in the probe is invented.
SEED_PAIRS: list[tuple[str, str]] = [
    ("will be blocked", "block ho jayega"),
    ("will be closed", "band ho jayega"),
    ("will be deactivated", "band ho jayegi"),
    ("is pending", "pending hai"),
    ("is incomplete", "adhoori hai"),
    ("right now", "abhi"),
    ("click here", "yahan click karein"),
    ("log in", "login karein"),
    ("to avoid", "se bachne ke liye"),
    ("in order to", "ke liye"),
    ("do not share", "share na karein"),
    ("thank you", "dhanyawad"),
    ("dear customer", "priya grahak"),
    ("your account", "aapka account"),
    ("your details", "aapki details"),
    ("your number", "aapka number"),
    ("your sim", "aapki SIM"),
    ("your", "aapka"),
    ("you", "aap"),
    ("we", "hum"),
    ("our", "hamara"),
    ("is", "hai"),
    ("are", "hain"),
    ("not", "nahi"),
    ("and", "aur"),
    ("or", "ya"),
    ("with", "ke saath"),
    ("without", "bina"),
    ("from", "se"),
    ("for", "ke liye"),
    ("today", "aaj"),
    ("now", "abhi"),
    ("immediately", "turant"),
    ("urgent", "zaroori"),
    ("please", "kripya"),
    ("kindly", "kripya"),
    ("send", "bhejein"),
    ("visit", "jaayein"),
    ("blocked", "block"),
    ("suspend", "band"),
    ("closed", "band"),
    ("complete", "poora"),
    ("incomplete", "adhoori"),
    ("failed", "fail"),
    ("received", "mila"),
    ("money", "paisa"),
    ("information", "jaankari"),
    ("customer", "grahak"),
    ("service", "service"),
    ("notice", "soochna"),
    ("last", "aakhri"),
    ("final", "aakhri"),
    ("prize", "inaam"),
    ("arrest", "giraftar"),
    ("help", "madad"),
    ("regards", "dhanyawad"),
    ("limit", "limit"),
    ("safe", "safe"),
    ("secure", "surakshit"),
    ("security", "security"),
    ("hours", "ghante"),
    ("minutes", "minute"),
    ("days", "din"),
]


def substitution_pairs() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(pairs used, seed pairs dropped because the form is absent).

    Pool phrases win over seed entries for the same English text, so a
    component's own wording is never replaced by something we wrote.
    """
    pool = " ".join(c["text"]["roman_hinglish"] for c in _component_pools()).lower()
    merged: dict[str, str] = {en: hi for en, hi in phrase_dictionary()}
    dropped: list[tuple[str, str]] = []
    for en, hi in SEED_PAIRS:
        if hi.lower() not in pool:
            dropped.append((en, hi))            # not in the pools -> not used
            continue
        merged.setdefault(en.lower(), hi)
    return sorted(merged.items(), key=lambda kv: -len(kv[0])), dropped


def make_substituter(pairs: list[tuple[str, str]]):
    """One longest-match-first regex over the whole message.

    Longest first matters: 'your account' must win before 'your', otherwise
    the output is word salad. A single pass also means a replacement can never
    be re-matched, so the rewrite cannot cascade into nonsense.
    """
    mapping = {en.lower(): hi for en, hi in pairs}
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Za-z])",
        re.IGNORECASE)

    def substitute(text: str) -> str:
        return pattern.sub(lambda m: mapping[m.group(0).lower()], text)

    return substitute


def to_roman_hinglish(text: str) -> str:
    """Rewrite one message with the seeded phrase dictionary."""
    pairs, _ = substitution_pairs()
    return make_substituter(pairs)(text)


HINGLISH_PROMPT = """Rewrite this Indian scam message in natural romanised \
Hindi-English (Hinglish), the way people actually type on WhatsApp. Keep the \
meaning and the ask identical, keep <URL>, <PHONE> and <UPI> exactly as they \
are, never add a real link or number, and output ONLY the rewritten message.

MESSAGE:
{message}
"""


def _hinglish_cache_dir() -> Path:
    folder = data_path("llm_cache_dir") / "hinglish"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def gemini_hinglish(texts: list[str]) -> tuple[dict[str, str], str]:
    """Optional: translate via Gemini, cached per source text.

    Returns (mapping, status). Without a key this is a clean skip - the probe
    then rests on the dictionary path only, and says so.
    """
    from src.synth.llm_paraphrase import (
        check_paraphrase, have_api_key, make_client)

    key = have_api_key()
    if not key:
        return {}, "no GEMINI_API_KEY - Gemini paraphrase skipped"
    client = make_client(key)
    if client is None:
        return {}, "google-genai unavailable - Gemini paraphrase skipped"

    folder = _hinglish_cache_dir()
    out: dict[str, str] = {}
    calls = 0
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = folder / f"{digest}.json"
        if path.exists():
            try:
                out[text] = json.loads(path.read_text(encoding="utf-8"))["text"]
                continue
            except Exception:
                pass
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=HINGLISH_PROMPT.format(message=text))
            new = (getattr(response, "text", "") or "").strip()
        except Exception as exc:                       # network/quota: stop
            return out, f"Gemini stopped after {calls} calls: {type(exc).__name__}"
        calls += 1
        if check_paraphrase(text, new) is not None:
            continue                                    # drop a bad rewrite
        path.write_text(json.dumps({"source": text, "text": new},
                                   ensure_ascii=False), encoding="utf-8")
        out[text] = new
    return out, f"Gemini rewrote {len(out)}/{len(texts)} messages ({calls} API calls)"


# ---------------------------------------------------------------------------
# (c) SAFE-CHAT PROBE - fresh legitimate messages, components unused in training
#
# These pools are written here, not in src/synth/components.py, so no training
# family can contain them; assert_probe_isolation() proves it against the table.
# ---------------------------------------------------------------------------
CHAT_OPENERS = [
    ("english", "Hey, quick one -"), ("english", "Hi both,"),
    ("english", "Just checking,"), ("english", "Morning!"),
    ("roman_hinglish", "Sun na,"), ("roman_hinglish", "Arre yaar,"),
    ("roman_hinglish", "Ek baat batani thi,"), ("roman_hinglish", "Hi, sab badhiya?"),
]
CHAT_BODIES = {
    "family": [
        ("english", "are we still doing dinner at home on Sunday or should I book a table"),
        ("english", "dad's medicine refill is done, I picked it up from the chemist near the market"),
        ("english", "I'll be late by twenty minutes, start without me and keep my plate aside"),
        ("roman_hinglish", "shaam ko ghar aa raha hoon, kuch laana hai kya market se"),
        ("roman_hinglish", "mummy ka phone charge pe laga do, unka call aayega shaam ko"),
        ("roman_hinglish", "bhai kal subah cricket khelne jaana hai, ground 7 baje"),
        ("english", "the wedding photos are on the shared drive, download before Friday"),
        ("roman_hinglish", "chachi ke ghar ka address bhej de, kal jaana hai"),
    ],
    "delivery": [
        ("english", "your parcel is at the security desk, the guard has it since 4 pm"),
        ("english", "the courier called, he will retry tomorrow between 11 and 2"),
        ("english", "order 4471 reached the local hub, delivery scheduled for tomorrow morning"),
        ("roman_hinglish", "parcel neeche guard ke paas rakh diya hai, shaam ko le lena"),
        ("roman_hinglish", "courier wala aaya tha, tum ghar pe nahi the, kal aayega"),
        ("english", "I kept the grocery bag inside the fridge, milk is in the door shelf"),
        ("roman_hinglish", "amazon ka box ghar ke bahar mat chhodna, andar le lo"),
        ("english", "the plumber is coming at 10, please be home to let him in"),
    ],
    "bank_alert": [
        ("english", "your monthly statement for the savings account is ready in the app, no action needed"),
        ("english", "we have updated the branch timings, the counter now opens at 9:30 am"),
        ("english", "your fixed deposit matures next month, the renewal form is at the branch"),
        ("english", "cheque book request received, it will be delivered to your registered address"),
        ("roman_hinglish", "aapke account ka statement app mein aa gaya hai, koi kaam nahi karna"),
        ("roman_hinglish", "branch ka time badal gaya hai, ab 9:30 baje khulta hai"),
        ("english", "the passbook update can be done at any kiosk, no token needed"),
        ("roman_hinglish", "FD agle mahine mature ho rahi hai, form branch mein hai"),
    ],
}
CHAT_TAILS = [
    ("english", "no rush, reply whenever."),
    ("english", "thanks!"),
    ("english", "let me know either way."),
    ("roman_hinglish", "theek hai, bata dena."),
    ("roman_hinglish", "chal, milte hain."),
    ("english", "see you soon."),
    ("roman_hinglish", "koi jaldi nahi hai."),
    ("english", "all good otherwise."),
]
CHAT_NAMES = ["Rohit", "Meera", "Sana", "Arjun", "Priya", "Vikram", "Neha", "Imran"]


def load_or_freeze_safe_chat(n: int, rng) -> pd.DataFrame:
    """The safe-chat test probe: FROZEN after the first build.

    Phase 5 tunes the model against this probe, so it must not drift when the
    generator changes. The first run writes
    data/processed/safe_chat_probe_frozen.parquet; every later run reads those
    rows exactly, and the file is never regenerated silently.
    """
    path = data_path("safe_chat_probe")
    if path.exists():
        df = pd.read_parquet(path)
        if len(df) == n:
            print(f"  safe-chat probe: FROZEN set of {len(df)} rows from {path.name}")
            return df
        print(f"  WARNING: frozen probe has {len(df)} rows, expected {n}; "
              "using the frozen rows anyway (they are the test set).")
        return df
    df = build_safe_chat_probe(n, rng)
    df.to_parquet(path, index=False)
    print(f"  safe-chat probe: froze {len(df)} rows to {path.name} "
          "(this is now the fixed test probe)")
    return df


def build_safe_chat_probe(n: int, rng) -> pd.DataFrame:
    """`n` fresh legitimate messages, half English and half roman Hinglish."""
    rows: list[dict] = []
    seen: set[str] = set()
    kinds = list(CHAT_BODIES)
    guard = 0
    while len(rows) < n and guard < n * 200:
        guard += 1
        kind = str(rng.choice(kinds))
        style_wanted = "english" if len(rows) % 2 == 0 else "roman_hinglish"
        openers = [o[1] for o in CHAT_OPENERS if o[0] == style_wanted]
        bodies = [b[1] for b in CHAT_BODIES[kind] if b[0] == style_wanted]
        tails = [t[1] for t in CHAT_TAILS if t[0] == style_wanted]
        parts = [str(rng.choice(openers))]
        parts.append(str(rng.choice(bodies)))
        if rng.random() < 0.6:
            parts.append(str(rng.choice(tails)))
        if rng.random() < 0.35:
            parts.append(f"- {rng.choice(CHAT_NAMES)}")
        text = " ".join(p for p in parts if p).strip()
        if text in seen:
            continue
        seen.add(text)
        rows.append({
            "id": f"probe_safe_chat_{len(rows) + 1:04d}",
            "text": text,
            "verdict": "safe",
            "tactics": [],
            "channel": "sms",
            "script_style": style_wanted,
            "date_received": "",
            "source": f"{PROBE_PREFIX}safe_chat",
            "template_family": "",
            "labeler": "probe_generator",
            "label_confidence": "high",
            "url_type": "none",
            "had_phone_number": False,
            "template_cluster": -1,
            "split": "probe",
            "real_stratum": "",
            "notes": f"fresh safe-chat probe ({kind}); components unused in training",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# frames + scoring
# ---------------------------------------------------------------------------
def probe_seed_scams() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two held-out scam pools every probe starts from."""
    df = load_messages()
    real = df[(df["split"] == "test_real")
              & (df["real_stratum"] == "external_scam")].reset_index(drop=True)
    if len(real) > N_ATTACK_REAL:
        real = real.sample(n=N_ATTACK_REAL, random_state=SEED).reset_index(drop=True)
    synth = df[(df["split"] == "test_b")
               & (df["verdict"] == "scam")].reset_index(drop=True)
    return real, synth


def _recall_at(scores: np.ndarray, t_high: float) -> float | None:
    if len(scores) == 0:
        return None
    return float((scores >= t_high).mean())


def _row(df: pd.DataFrame, source: str, extra: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    out["source"] = source
    out["split"] = "probe"
    out["real_stratum"] = ""
    if extra:
        for key, value in extra.items():
            out[key] = value
    return out


def run_attacks(predictor, temperature: float, t_high: float,
                rng) -> tuple[list[dict], list[pd.DataFrame]]:
    """Recall at the fixed point for every attack, per source pool."""
    real, synth = probe_seed_scams()
    pools = [("real_scams", real), ("test_b_scams", synth)]
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    clean_scores: dict[str, np.ndarray] = {}

    for pool_name, pool in pools:
        if pool.empty:
            continue
        clean = apply_temperature(
            predictor.scores(pool), temperature)
        clean_scores[pool_name] = clean
        rows.append({
            "pool": pool_name, "attack": "clean (baseline)", "n": int(len(pool)),
            "recall": _recall_at(clean, t_high),
            "mean_p": float(clean.mean()),
            "delta_recall": 0.0,
            "url_rows_affected": int(pool["text"].astype(str).str.contains(
                r"<URL>|https?://|www\.", regex=True).sum()),
        })

    for name, label, fn in ATTACKS:
        for pool_name, pool in pools:
            if pool.empty:
                continue
            texts = [fn(str(t), rng) for t in pool["text"]]
            changed = np.array([a.strip() != b.strip()
                                for a, b in zip(pool["text"].astype(str), texts)])
            cal = apply_temperature(predictor.scores(
                pool.assign(text=texts)), temperature)
            base = _recall_at(clean_scores[pool_name], t_high)
            recall = _recall_at(cal, t_high)
            # Recall on the rows the attack ACTUALLY edited: an attack that is
            # a no-op on a row (no URL to obfuscate) cannot hurt that row, and
            # letting those rows dilute the number would flatter the model.
            recall_changed = (_recall_at(cal[changed], t_high)
                              if changed.any() else None)
            rows.append({
                "pool": pool_name, "attack": name, "label": label,
                "n": int(len(pool)), "n_changed": int(changed.sum()),
                "recall": recall, "recall_changed": recall_changed,
                "mean_p": float(cal.mean()),
                "delta_recall": (None if recall is None or base is None
                                 else recall - base),
                "delta_recall_changed": (None if recall_changed is None or base is None
                                         else recall_changed - base),
            })
            probe = _row(pool.assign(text=texts,
                                     id=pool["id"].astype(str) + f"_{name}"),
                         f"{PROBE_PREFIX}attack_{name}",
                         {"probe_kind": f"attack_{name}",
                          "probe_pool": pool_name,
                          "clean_text": list(pool["text"].astype(str))})
            frames.append(probe)
    return rows, frames


def run_transfer(predictor, temperature: float, t_high: float,
                 rng) -> tuple[dict, pd.DataFrame | None]:
    """English -> roman Hinglish transfer probe on held-out real scams."""
    real, _ = probe_seed_scams()
    if real.empty:
        return {"available": False, "reason": "no held-out real scams"}, None
    sample = real.sample(n=min(N_TRANSFER, len(real)),
                         random_state=SEED).reset_index(drop=True)

    phrases, dropped = substitution_pairs()
    substitute = make_substituter(phrases)
    source_en = sample["text"].astype(str).tolist()
    hinglish = [substitute(t) for t in source_en]

    changed = sum(1 for a, b in zip(source_en, hinglish) if a.strip() != b.strip())
    token_share = float(np.mean([
        sum(1 for a, b in zip(_words(x), _words(y)) if a.lower() != b.lower())
        / max(len(_words(x)), 1)
        for x, y in zip(source_en, hinglish)]))

    en_scores = apply_temperature(predictor.scores(sample), temperature)
    hi_frame = sample.assign(text=hinglish)
    hi_scores = apply_temperature(predictor.scores(hi_frame), temperature)

    gemini_map, gemini_status = gemini_hinglish(source_en[:REVIEW_N])
    gemini_scores = None
    if gemini_map:
        order = [t for t in source_en if t in gemini_map]
        g_frame = sample[sample["text"].astype(str).isin(order)]
        g_texts = [gemini_map[t] for t in g_frame["text"].astype(str)]
        gemini_scores = apply_temperature(
            predictor.scores(g_frame.assign(text=g_texts)), temperature)

    result = {
        "available": True,
        "n": int(len(sample)),
        "dictionary_phrases": len(phrases),
        "seed_pairs_dropped": [f"{en} -> {hi}" for en, hi in dropped],
        "rows_changed": int(changed),
        "share_changed": float(changed / max(len(sample), 1)),
        "mean_token_replacement": token_share,
        "recall_english": _recall_at(en_scores, t_high),
        "recall_hinglish": _recall_at(hi_scores, t_high),
        "mean_p_english": float(en_scores.mean()),
        "mean_p_hinglish": float(hi_scores.mean()),
        "gemini_status": gemini_status,
        "gemini_n": int(len(gemini_scores)) if gemini_scores is not None else 0,
        "recall_gemini": (_recall_at(gemini_scores, t_high)
                          if gemini_scores is not None else None),
        # PRIVACY: the JSON artefact must not carry probe text either, so only
        # the shape (word count) is stored. Scores stay, which is what the
        # report actually argues from.
        "examples": [{"english": f"[withheld {len(str(a).split())} words]",
                      "hinglish": f"[withheld {len(str(b).split())} words]",
                      "p_english": float(pe), "p_hinglish": float(ph)}
                     for a, b, pe, ph in zip(source_en, hinglish,
                                             en_scores, hi_scores)
                     if a.strip() != b.strip()][:5],
    }
    frame = _row(hi_frame.assign(
        id=sample["id"].astype(str) + "_hinglish"),
        f"{PROBE_PREFIX}hinglish_transfer",
        {"probe_kind": "hinglish_transfer",
         "probe_pool": "real_scams",
         "clean_text": list(sample["text"].astype(str)),
         "notes": "translation-derived probe, not real Hinglish"})
    return result, frame


def run_safe_chat(predictor, temperature: float, t_high: float,
                  rng) -> tuple[dict, pd.DataFrame]:
    """False-positive rate on 200 fresh legitimate messages."""
    probe = load_or_freeze_safe_chat(N_SAFE_CHAT, rng)
    cal = apply_temperature(predictor.scores(probe), temperature)
    flagged = cal >= t_high
    train = load_messages()
    train_texts = train.loc[train["split"] == "train", "text"].astype(str).tolist()
    sim = max_similarity_to_train(train_texts, probe["text"].astype(str).tolist())
    probe["max_train_sim"] = sim
    frame = _row(probe, f"{PROBE_PREFIX}safe_chat",
                 {"probe_kind": "safe_chat",
                  "probe_pool": "fresh_safe_messages"})

    by_style = {}
    for style, sub_idx in probe.groupby("script_style").groups.items():
        idx = list(sub_idx)
        by_style[str(style)] = {
            "n": len(idx),
            "fpr": float(flagged[idx].mean()),
            "mean_p": float(cal[idx].mean()),
            "too_small": too_small(len(idx)),
        }
    result = {
        "n": int(len(probe)),
        "n_flagged": int(flagged.sum()),
        "fpr": float(flagged.mean()),
        "mean_p": float(cal.mean()),
        "max_train_sim": float(np.max(sim)) if len(sim) else None,
        "mean_train_sim": float(np.mean(sim)) if len(sim) else None,
        "by_style": by_style,
        # PRIVACY: shapes only - never the probe text itself.
        "examples_flagged": [
            f"[withheld {len(str(t).split())} words / {len(str(t))} chars]"
            for t in probe.loc[flagged, "text"].head(5).tolist()],
        "too_small_for_claims": too_small(len(probe)),
        "frozen": True,
        "source_file": str(data_path("safe_chat_probe")),
    }
    return result, frame


def assert_probe_isolation(probes: pd.DataFrame) -> None:
    """Hard rule: no probe row may exist in the training table."""
    if probes.empty:
        return
    bad_source = probes.loc[~probes["source"].astype(str).str.startswith(PROBE_PREFIX)]
    if len(bad_source):
        raise AssertionError(
            f"{len(bad_source)} probe rows do not carry a source=probe_* tag")
    # Only the TRAIN split matters: a probe may legitimately reproduce a
    # held-out test message (an attack that changes nothing on that row), but
    # it may never coincide with a row the model learned from.
    train = load_messages()
    train = train[train["split"] == "train"]
    train_ids = set(train["id"].astype(str))
    train_texts = set(train["text"].astype(str).str.strip().str.lower())
    leaked = probes[probes["id"].astype(str).isin(train_ids)
                    | probes["text"].astype(str).str.strip().str.lower().isin(train_texts)]
    if len(leaked):
        raise AssertionError(
            f"{len(leaked)} probe rows are already in messages.parquet - "
            "probes must never be training data")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_all(name: str) -> dict:
    """Every probe, scored at the fixed operating point."""
    point = load_operating_point(name)
    temperature = float(point["calibration"]["temperature"])
    t_high = float(point["operating_point"]["t_high"])
    predictor = load_predictor(name)
    # One stream per probe: editing the attack code must not change which safe
    # messages the safe-chat probe draws, and vice versa.
    rng_attacks = np.random.default_rng(SEED)
    rng_transfer = np.random.default_rng(SEED + 1)
    rng_chat = np.random.default_rng(SEED + 2)

    print(f"\n(src/robustness.py) model={name}  fixed point: "
          f"t_high={t_high:.4f} temperature={temperature:.4f}")
    print("  robustness probe, not a real-world claim")

    attacks, attack_frames = run_attacks(predictor, temperature, t_high,
                                         rng_attacks)
    transfer, transfer_frame = run_transfer(predictor, temperature, t_high,
                                            rng_transfer)
    safe_chat, safe_frame = run_safe_chat(predictor, temperature, t_high,
                                          rng_chat)

    frames = attack_frames + ([transfer_frame] if transfer_frame is not None else [])
    frames.append(safe_frame)
    probes = pd.concat(frames, ignore_index=True)
    assert_probe_isolation(probes)
    out = data_path("probes_parquet")
    probes.to_parquet(out, index=False)
    print(f"  WROTE {out} ({len(probes):,} probe rows, source=probe_*)")

    result = {
        "model": name,
        "t_high": t_high,
        "temperature": temperature,
        "label": "robustness probe, not a real-world claim",
        "attacks": attacks,
        "transfer": transfer,
        "safe_chat": safe_chat,
        "probe_rows_written": int(len(probes)),
        "probe_max_train_sim": safe_chat["max_train_sim"],
    }
    worst = sorted(
        [r for r in attacks if r["attack"] != "clean (baseline)"
         and r["delta_recall_changed"] is not None],
        key=lambda r: r["delta_recall_changed"])[:3]
    for row in worst:
        print(f"  worst attack: {row['pool']:14s} {row['attack']:20s} "
              f"n={row['n']} edited={row['n_changed']} "
              f"recall(all)={row['recall']:.3f} "
              f"recall(edited)={row['recall_changed']:.3f} "
              f"(delta {row['delta_recall_changed']:+.3f})")
    if transfer.get("available"):
        print(f"  transfer: English recall {transfer['recall_english']:.3f} -> "
              f"Hinglish-derived {transfer['recall_hinglish']:.3f} "
              f"(n={transfer['n']}, {transfer['share_changed']:.0%} of rows changed)")
        print(f"  {transfer['gemini_status']}")
    print(f"  safe chat: FPR {safe_chat['fpr']:.3f} on n={safe_chat['n']} fresh "
          f"legitimate messages")
    return result


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def markdown(result: dict) -> list[str]:
    """All three probes as report tables."""
    lines = [
        "## Robustness probes",
        "",
        "> **Robustness probe, not a real-world claim.** Every row below was "
        "produced by a script in `src/robustness.py`, not by an adversary in "
        "the wild. The numbers say how a fixed operating point survives "
        "mechanical edits; they say nothing about next quarter's scams.",
        "",
        f"Fixed operating point: `t_high` = **{result['t_high']:.3f}**, "
        f"temperature = {result['temperature']:.3f} (fitted on validation "
        "only). A scam counts as caught when its calibrated probability "
        "clears `t_high`; the abstain band is empty for this model, so "
        "`suspicious` cannot rescue recall.",
        "",
        "### (a) Perturbation attacks - scam recall at the fixed point",
        "",
        "| pool | attack | n | rows edited | recall (all) | recall (edited) "
        "| delta vs clean | mean p | note |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in result["attacks"]:
        note = claim_flag(row["n"])
        delta = row["delta_recall"]
        if row["attack"] == "clean (baseline)" or delta is None:
            delta_text = "baseline" if row["attack"] == "clean (baseline)" else "n/a"
        else:
            delta_text = f"{delta:+.3f}"
        lines.append(
            f"| `{row['pool']}` | {row['attack']} | {row['n']:,} | "
            f"{row.get('n_changed', row['n']):,} | {_fmt(row['recall'])} | "
            f"{_fmt(row.get('recall_changed'))} | {delta_text} | "
            f"{_fmt(row['mean_p'])} | {note} |")
    lines += [
        "",
        "> Attack rates: typos swap two characters in 20% of words; spaces are "
        "inserted inside 20% of words; lookalikes replace one character in 30% "
        "of words; leetspeak rewrites 25% of words; emoji inserts 2-4 symbols; "
        "URL obfuscation writes `hxxps://site[.]tld` (a scrubbed `<URL>` is "
        "replaced by the same obfuscated form).",
        "",
    ]

    transfer = result["transfer"]
    lines += ["### (b) Hinglish transfer probe - **translation-derived, not real Hinglish**",
              ""]
    if not transfer.get("available"):
        lines += [f"(skipped: {transfer.get('reason')})", ""]
    else:
        lines += [
            f"{transfer['n']} held-out real English scams were rewritten with a "
            f"phrase dictionary of {transfer['dictionary_phrases']} pairs built "
            "from the `roman_hinglish` pools in `src/synth/components.py` plus "
            "a hand-aligned seed whose every Hinglish form is verified to "
            f"appear in those pools ({len(transfer['seed_pairs_dropped'])} "
            "seed pairs were dropped for failing that check).",
            "",
            "| version | n | scam recall | mean p |",
            "|---|---:|---|---|",
            f"| English (original) | {transfer['n']} | "
            f"{_fmt(transfer['recall_english'])} | {_fmt(transfer['mean_p_english'])} |",
            f"| roman Hinglish (dictionary rewrite) | {transfer['n']} | "
            f"{_fmt(transfer['recall_hinglish'])} | {_fmt(transfer['mean_p_hinglish'])} |",
        ]
        if transfer["gemini_n"]:
            lines.append(
                f"| roman Hinglish (Gemini paraphrase of {transfer['gemini_n']}) | "
                f"{transfer['gemini_n']} | {_fmt(transfer['recall_gemini'])} | n/a |")
        lines += [
            "",
            f"- {transfer['share_changed']:.1%} of rows changed; on average "
            f"{transfer['mean_token_replacement']:.1%} of a message's words were "
            "replaced, so these are light rewrites, not full translations.",
            f"- Gemini path: {transfer['gemini_status']}.",
            "",
        ]
        if transfer["examples"]:
            lines += ["Examples for manual review (English -> derived Hinglish):", ""]
            for ex in transfer["examples"]:
                # PRIVACY: scores only. The probe text itself is withheld.
                lines += [f"> **EN** {ex['p_english']:.3f} -> **HI** "
                          f"{ex['p_hinglish']:.3f} {ex['english']}", ""]

    safe = result["safe_chat"]
    lines += ["### (c) Fresh safe-chat probe - false positive rate", "",
              "| style | n | flagged as scam | FPR | mean p | note |",
              "|---|---:|---:|---|---|---|"]
    for style, row in sorted(safe["by_style"].items()):
        lines.append(
            f"| {style} | {row['n']} | - | {_fmt(row['fpr'])} | "
            f"{_fmt(row['mean_p'])} | {claim_flag(row['n'])} |")
    lines += [
        f"| **all** | {safe['n']} | {safe['n_flagged']} | {_fmt(safe['fpr'])} | "
        f"{_fmt(safe['mean_p'])} | {claim_flag(safe['n'])} |",
        "",
        f"Novelty check: the highest similarity of any probe message to a "
        f"training row is **{_fmt(safe['max_train_sim'], 2)}** (mean "
        f"{_fmt(safe['mean_train_sim'], 2)}), so these are not near-copies of "
        "training text.",
        "",
    ]
    if safe["examples_flagged"]:
        lines += ["Safe messages the model calls scam:", ""]
        # PRIVACY: the list already holds shape descriptors, never text.
        lines += [f"> flagged probe: {text}" for text in safe["examples_flagged"]]
        lines.append("")
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: robustness probes (never trained on)")
    parser.add_argument("--model", default="verdict_lr_plus")
    args = parser.parse_args(argv)

    result = run_all(args.model)
    out_md = data_path("robustness_md")
    out_md.write_text("\n".join(markdown(result)), encoding="utf-8")
    out_json = data_path("results_json").parent / "robustness.json"
    out_json.write_text(json.dumps(result, indent=2, default=str),
                        encoding="utf-8")
    print(f"  WROTE {out_md}")
    print(f"  WROTE {out_json}")
    return result


if __name__ == "__main__":
    main()
