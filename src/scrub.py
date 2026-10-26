"""
src/scrub.py - remove personal information (PII) from a message.

WHY THIS EXISTS
---------------
Raw messages from real people can contain phone numbers, UPI ids, email
addresses, account numbers and names. We must never store those in
data/interim, data/processed or data/golden. This module rewrites them into
placeholder tokens like <PHONE> while remembering useful *structure*
(did it contain a phone number? what kind of URL?) because that structure is
itself a scam signal.

HONEST WARNING - READ THIS
--------------------------
Regex scrubbing is IMPERFECT. It cannot catch every format, every language,
or a name written in the middle of a sentence. After running the pipeline you
MUST manually skim every scrubbed message in data/interim/ before it is kept.
This module is a safety net, not a guarantee.

Typical use:
    from src.scrub import scrub
    clean, meta = scrub("Call me on 98765 43210")
    # clean -> "Call me on <PHONE>"
    # meta   -> {"had_phone_number": True, "url_type": "none", ...}
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from src.config import BRAND_TOKENS, NAME_STOPWORDS, ODD_TLDS, OFFICIAL_TLDS, SHORTENER_DOMAINS

# ---------------------------------------------------------------------------
# REGULAR EXPRESSIONS  (each one is explained in plain English)
# ---------------------------------------------------------------------------

# Email address: something@something.tld  (the domain part must contain a dot,
# which is what separates it from a UPI id like name@okhdfcbank).
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w\-]+(?:\.[\w\-]+)+")

# Any URL: starts with http(s):// or www., OR looks like domain.tld(/path).
# We process URLs BEFORE bare-digit patterns so "http://1.2.3.4/a91..." is
# classified as a URL and not mistaken for a phone number.
URL_RE = re.compile(
    r"(?i)"
    r"(?:https?://|www\.)\S+"          # explicit http://... or www....
    r"|\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)+(?::\d+)?(?:/\S*)?"  # example.com/xyz
)

# UPI id: name@bankhandle with NO dot in the handle (name@okaxis, a.b@paytm).
# Must run BEFORE nothing else - emails already replaced by this point.
# The (?!.\w) part allows a trailing sentence period: "...@okhdfcbank." still
# counts as a UPI id, but "name@example.com" does not (dot followed by a word).
UPI_RE = re.compile(r"(?<![\w.\-])[a-z0-9._\-]{2,}@[a-z]{2,14}(?![\w\-])(?!\.\w)", re.IGNORECASE)

# OTP: a 4-8 digit code that sits NEXT TO a word like OTP / code / PIN.
# (?!\d) after the digits stops it from eating the start of a 10-digit phone.
OTP_FWD_RE = re.compile(
    r"(?i)"
    r"\b(?:otp|one[\s\-]?time[\s\-]?password|verification[\s\-]?code|"
    r"login[\s\-]?code|security[\s\-]?code|auth(?:entication)?[\s\-]?code|"
    r"passcode|code|pin)\b"
    r"[^\d]{0,20}"
    r"(?<!\d)(\d{4,8})(?!\d)"
)

# Same idea with the digits BEFORE the word: "482913 is your OTP".
OTP_REV_RE = re.compile(
    r"(?i)"
    r"(?<!\d)(\d{4,8})(?!\d)"
    r"[^\d]{0,15}"
    r"\b(?:otp|code|pin)\b"
)

# Indian phone numbers in many formats:
#   9876543210 | +91 98765 43210 | 91-98765-43210 | 09876543210
#   | (+91) 98765 43210 | 9 8 7 6 5 4 3 2 1 0
# (?<!\d) and (?!\d) stop it from matching in the middle of a LONGER digit run
# (a 16-digit card number must fall through to <ACCT> instead).
PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?91[\s\-\)]*)?"   # optional country code: +91 or 91, maybe ')'
    r"(?:0[\s\-]*)?"         # optional trunk prefix: 0
    r"[6-9](?:[\s\-\.]?\d){9}"   # 10 digits total, starting 6-9,
    r"(?!\d)"                # separators allowed between any two digits
)

# Long digit run of 9+ digits: account number, card number, customer id.
ACCT_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")

# Person-name heuristic: a capitalized word right after a title
# (Dear/Mr/Mrs/Shri/Smt/Sir/Madam) or after "I am" / "my name is" / "This is".
# The [A-Z] on the NAME stays case-sensitive on purpose: "This is urgent" must
# NOT become "This is <NAME>". Only the prefix words get capital variants.
# IMPERFECT ON PURPOSE - we would rather replace a few innocent words with
# <NAME> than leak a real name. Manual skim still required.
NAME_RE = re.compile(
    r"\b(?:[Dd]ear|[Ss]hri|[Ss]mt|[Mm]r|[Mm]rs|[Mm]s|[Mm]iss|[Ss]ir|[Mm]adam)"
    r"\s+([A-Z][a-zA-Z]{1,24})\b"
    r"|\b(?:[Ii] am|[Ii]'m|[Mm]y name is|[Tt]his is)\s+([A-Z][a-zA-Z]{1,24})\b"
)

# Sign-off heuristic (Phase 2 gap fix): "Thanks, <Name>" / "Regards, <Name>"
# / "Warm regards, <Name>" at the END of a message. Same conservative rule as
# NAME_RE: only a CAPITALISED word directly after a sign-off word is replaced,
# so "Thanks for the update" is left alone. The name is the only capture
# group, which is what _name_sub() uses to find it.
SIGNOFF_RE = re.compile(
    r"\b(?:[Tt]hanks|[Tt]hnx|[Rr]egards?|[Bb]est\s+[Rr]egards?"
    r"|[Ww]arm\s+[Rr]egards?|[Ww]ith\s+[Rr]egards?"
    r"|[Tt]hanks\s*(?:&|and|\+)\s*[Rr]egards"
    r"|[Cc]heers|[Ll]ots\s+of\s+love|[Ww]ith\s+love)"
    r"\s*(?:,|:|;|!|-)?\s+"          # separator: "Thanks, " or "Regards -"
    r"([A-Z][a-zA-Z]{1,24})\b"       # the sign-off name (must be capitalised)
)

# Leftover digit runs of 5+ digits that nothing above caught. We keep them
# (spec says 9+ becomes <ACCT>) but FLAG them so the manual skim knows where
# to look.
LEFTOVER_DIGIT_RE = re.compile(r"(?<!\d)\d{5,}(?!\d)")


# ---------------------------------------------------------------------------
# URL CLASSIFICATION - we lose the URL text but KEEP its type as a signal
# ---------------------------------------------------------------------------

def _url_flags(url: str) -> list[str]:
    """Return EVERY flag that applies to a URL (for the meta['url_flags'] list)."""
    raw = url.lower()
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    try:
        host = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return []
    # Remove a leading 'www.' as a PREFIX (not lstrip, which eats w's and dots
    # from hosts like 'win.example.com').
    if host.startswith("www."):
        host = host[4:]
    flags: list[str] = []
    if host in SHORTENER_DOMAINS or any(host.endswith("." + s) for s in SHORTENER_DOMAINS):
        flags.append("shortener")
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        flags.append("ip_address")
    if any(tok in host for tok in BRAND_TOKENS) or host.endswith((".gov.in", ".nic.in", ".ac.in")):
        flags.append("official_looking")
    tld = host.rsplit(".", 1)[-1]
    if tld in ODD_TLDS:
        flags.append("odd_tld")
    return flags


def classify_url(url: str) -> str:
    """Return one of: shortener, ip_address, odd_tld, official_looking, other.

    Priority (first match wins):
      1. shortener         - known shortener domain (bit.ly, tinyurl, ...)
      2. ip_address        - host is a raw IPv4 address (e.g. http://1.2.3.4/)
      3. official_looking  - host contains a bank/telecom/gov brand token
                             (hdfc-secure.com) or a government TLD (.gov.in)
      4. odd_tld           - suspicious TLD (.xyz, .tk, .top, ...)
      5. other             - everything else

    meta["url_flags"] keeps EVERY flag that matched, so nothing is lost even
    when the single url_type column must pick one.
    """
    flags = _url_flags(url)
    for f in ("shortener", "ip_address", "official_looking", "odd_tld"):
        if f in flags:
            return f
    return "other"


# TLDs we recognise on a BARE domain (no http://). This whitelist is what
# stops ordinary dotted text ("john.doe", "Rs.50", "kya.aap.thik.ho") from
# being mistaken for a URL and replaced with <URL>.
_KNOWN_TLDS = set(OFFICIAL_TLDS) | set(ODD_TLDS)


def _looks_like_url(url: str) -> bool:
    """Decide whether a URL_RE match is really a URL (vs dotted prose).

    Explicit http(s):// or www. always count. A bare domain only counts when
    its last label is a TLD we know, or its host is a known shortener.
    """
    u = url.lower()
    if u.startswith(("http://", "https://", "www.")):
        return True
    host_part = u.split("/")[0].split(":")[0]   # strip any path/port
    if host_part in SHORTENER_DOMAINS or any(host_part.endswith("." + s) for s in SHORTENER_DOMAINS):
        return True
    tld = host_part.rsplit(".", 1)[-1]
    return tld in _KNOWN_TLDS


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def scrub(text: str) -> tuple[str, dict]:
    """Replace PII in `text` with placeholder tokens.

    Returns (clean_text, meta) where meta records:
        had_phone_number : bool - a phone-number pattern was found
        url_type         : str  - one of none/shortener/ip_address/odd_tld/
                                  official_looking/other (FIRST url wins)
        url_flags        : list - every url flag found across all urls
        n_urls           : int
        n_emails, n_upi, n_otps, n_accts, n_names : counts of each replacement
        possible_name    : bool - a name-like word was replaced
        leftover_digits  : list - 5+ digit runs nothing caught (SKIM THESE!)

    NOTE: scrubbing is imperfect. Always manually skim the output.
    """
    if text is None:
        text = ""
    out = str(text)

    meta = {
        "had_phone_number": False,
        "url_type": "none",
        "url_flags": [],
        "n_urls": 0,
        "n_emails": 0,
        "n_upi": 0,
        "n_otps": 0,
        "n_accts": 0,
        "n_names": 0,
        "possible_name": False,
        "leftover_digits": [],
    }

    # -- STEP 1: emails first, so their domains are not mistaken for URLs ----
    out, n = EMAIL_RE.subn("<EMAIL>", out)
    meta["n_emails"] = n

    # -- STEP 2: UPI ids (name@bankhandle) BEFORE URLs, so a dotted handle
    # like "john.doe@paytm" is not mistaken for the domain "john.doe". --------
    out, meta["n_upi"] = UPI_RE.subn("<UPI>", out)

    # -- STEP 3: URLs - classify BEFORE replacing so we keep url_type -------
    url_types: list[str] = []
    for m in URL_RE.finditer(out):
        url = m.group(0)
        if not _looks_like_url(url):
            continue  # dotted prose like "Rs.50" or "john.doe"
        url_types.append(classify_url(url))
        meta["url_flags"].extend(_url_flags(url))
    meta["n_urls"] = len(url_types)
    if url_types:
        # url_type column keeps the FIRST url's type; flags keep everything.
        meta["url_type"] = url_types[0]
    out = URL_RE.sub(_url_replacer, out)

    # -- STEP 4: OTP codes next to OTP/code/PIN words (before phone/acct!) --
    out, n1 = OTP_FWD_RE.subn("<OTP>", out)
    out, n2 = OTP_REV_RE.subn("<OTP>", out)
    meta["n_otps"] = n1 + n2

    # -- STEP 5: phone numbers (before generic digit runs) ------------------
    def _phone_sub(m: re.Match) -> str:
        meta["had_phone_number"] = True
        return "<PHONE>"

    out = PHONE_RE.sub(_phone_sub, out)

    # -- STEP 6: long digit runs (account / card numbers) -------------------
    out, meta["n_accts"] = ACCT_RE.subn("<ACCT>", out)

    # -- STEP 7: person-name heuristic -------------------------------------
    def _name_sub(m: re.Match) -> str:
        # The NAME is whichever capture group matched: each name regex only
        # captures the name itself (titles/prefixes live in the non-capturing
        # part of the pattern), so the first non-None group is the name.
        idx = next((i for i, g in enumerate(m.groups(), start=1) if g), None)
        word = m.group(idx) if idx else ""
        # Do not scrub words that look like names but are not (Dear Team, ...)
        if word in NAME_STOPWORDS:
            return m.group(0)
        meta["n_names"] += 1
        meta["possible_name"] = True
        # Replace ONLY the name word, keeping the title/prefix intact.
        prefix = m.group(0)[: m.start(idx) - m.start(0)]
        return prefix + "<NAME>"

    out = NAME_RE.sub(_name_sub, out)
    # Phase 2 gap fix: catch sign-offs like "Thanks, Ravi" / "Regards, Priya".
    out = SIGNOFF_RE.sub(_name_sub, out)

    # -- STEP 8: flag any leftover 5+ digit runs for manual skim ------------
    meta["leftover_digits"] = sorted(set(LEFTOVER_DIGIT_RE.findall(out)))

    # Collapse any doubled spaces created by replacements (tidy output).
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out, meta


def _url_replacer(m: re.Match) -> str:
    """Replace a URL with <URL>, unless it is really dotted prose."""
    url = m.group(0)
    return url if not _looks_like_url(url) else "<URL>"


# ---------------------------------------------------------------------------
# SELF TEST  (run: python -m src.scrub)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = (
        "Dear Rohit, your OTP is 482913. Send Rs 1 to 98765 43210 or "
        "to raj@okhdfcbank. Click http://192.168.0.5/kyc or bit.ly/abc123. "
        "A/C 123456789012, mail me at r.k@example.com. I am Amit."
    )
    clean, info = scrub(demo)
    print("RAW  :", demo)
    print("CLEAN:", clean)
    print("META :")
    for k, v in info.items():
        print(f"   {k:18} = {v}")
