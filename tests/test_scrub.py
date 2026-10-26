"""
tests/test_scrub.py - the scrubber is our only privacy defense, so it gets
the most tests. Two goals:
  1. every documented format variant actually gets scrubbed
  2. scrubbed output never contains the original identifiers

Run:  python -m pytest tests/test_scrub.py -v
"""
import re

from src.scrub import classify_url, scrub

# ---------------------------------------------------------------------------
# 1. PHONE NUMBER FORMAT VARIANTS
# ---------------------------------------------------------------------------
PHONE_VARIANTS = [
    "9876543210",                  # plain 10 digits
    "+919876543210",               # +91 glued
    "+91 98765 43210",             # +91 with spaces
    "+91-98765-43210",             # +91 with dashes
    "919876543210",                # 91 prefix without +
    "91 98765 43210",              # 91 prefix with spaces
    "09876543210",                 # trunk prefix 0
    "98765-43210",                 # dash in the middle
    "98765 43210",                 # space in the middle
    "98765.43210",                 # dot in the middle
    "9 8 7 6 5 4 3 2 1 0",        # space between every digit
    "(+91) 98765 43210",           # parentheses around the code
]


def test_phone_variants_all_scrubbed():
    """Every documented Indian phone format must become <PHONE>."""
    for phone in PHONE_VARIANTS:
        text = f"Call me on {phone} please"
        clean, meta = scrub(text)
        assert "<PHONE>" in clean, f"phone not scrubbed: {phone!r} -> {clean!r}"
        assert meta["had_phone_number"] is True, f"meta flag missing for {phone!r}"


# ---------------------------------------------------------------------------
# 2. UPI IDS
# ---------------------------------------------------------------------------
UPI_VARIANTS = [
    "9876543210@okaxis",
    "rahul@okhdfcbank",
    "john.doe@paytm",
    "abc@ybl",
    "name.surname@upi",
    "shop99@ibl",
]


def test_upi_ids_scrubbed():
    for upi in UPI_VARIANTS:
        clean, meta = scrub(f"Pay to {upi} now")
        assert "<UPI>" in clean, f"UPI not scrubbed: {upi!r} -> {clean!r}"
        assert meta["n_upi"] >= 1


# ---------------------------------------------------------------------------
# 3. EMAILS
# ---------------------------------------------------------------------------
def test_emails_scrubbed():
    for email in ["ravi@example.com", "a.b+tag@sub.domain.co.in", "x_y@mail.in"]:
        clean, meta = scrub(f"Mail me at {email}")
        assert "<EMAIL>" in clean, f"email not scrubbed: {email!r} -> {clean!r}"
        assert meta["n_emails"] == 1


# ---------------------------------------------------------------------------
# 4. LONG DIGIT RUNS (account / card numbers)
# ---------------------------------------------------------------------------
def test_long_digit_runs_scrubbed():
    for digits in ["123456789012", "5555555555555555", "999988887777"]:
        clean, _ = scrub(f"A/C {digits} mein paisa hai")
        assert digits not in clean, f"account number leaked: {digits} -> {clean!r}"
        assert "<ACCT>" in clean


# ---------------------------------------------------------------------------
# 5. OTP CODES NEXT TO KEYWORDS
# ---------------------------------------------------------------------------
OTP_VARIANTS = [
    "your OTP is 482913",
    "OTP: 1234",
    "one time password 998877",
    "verification code 556677",
    "use code 445566 to login",
    "PIN 778899",
    "482913 is your OTP",       # digits BEFORE the keyword
]


def test_otp_codes_scrubbed():
    for phrase in OTP_VARIANTS:
        clean, meta = scrub(f"Hello, {phrase} ok")
        assert "<OTP>" in clean, f"OTP not scrubbed: {phrase!r} -> {clean!r}"


def test_otp_regex_does_not_eat_long_runs():
    """A 10-digit phone next to the word 'code' must NOT be eaten as an OTP."""
    clean, meta = scrub("your reference code 9876543210")
    assert "<OTP>" not in clean          # 10 digits is not an OTP length
    assert meta["had_phone_number"] is True  # it falls through to <PHONE>


# ---------------------------------------------------------------------------
# 6. URL TYPE DETECTION (we must not lose this signal)
# ---------------------------------------------------------------------------
def test_url_types():
    cases = {
        "visit bit.ly/abc123 now": "shortener",
        "tinyurl.com/xyz is here": "shortener",
        "open http://192.168.0.55/kyc": "ip_address",
        # sbi-kyc-update.xyz is BOTH brand-lookalike and odd TLD; the url_type
        # column keeps official_looking (higher priority) and url_flags keeps both.
        "click sbi-kyc-update.xyz/x": "official_looking",
        "claim your prize at lucky-prize.xyz/win": "odd_tld",
        "see hdfcsecure.com/offer": "official_looking",
        "go to google.com/mail": "other",
    }
    for text, expected in cases.items():
        clean, meta = scrub(text)
        assert meta["url_type"] == expected, f"{text!r}: got {meta['url_type']}, want {expected}"
        assert "<URL>" in clean, f"URL text not replaced: {text!r} -> {clean!r}"


def test_url_type_none_when_no_url():
    _, meta = scrub("koi link nahi hai is message mein")
    assert meta["url_type"] == "none"


def test_classify_url_direct():
    assert classify_url("https://www.bit.ly/x") == "shortener"
    assert classify_url("http://10.0.0.1/") == "ip_address"
    assert classify_url("cheap-prizes.top/a") == "odd_tld"
    assert classify_url("irctc-refund.in/x") == "official_looking"
    # brand token + odd tld: official_looking wins, but flags keep both
    assert classify_url("phonepe-verify.top/a") == "official_looking"


# ---------------------------------------------------------------------------
# 7. NAME HEURISTIC
# ---------------------------------------------------------------------------
def test_names_scrubbed():
    for text in ["Dear Rohit, ...", "Shri Sharma ji", "I am Amit speaking",
                 "my name is Sunita", "This is Priya"]:
        clean, meta = scrub(text)
        assert "<NAME>" in clean, f"name not scrubbed: {text!r} -> {clean!r}"


def test_signoff_names_scrubbed():
    """Phase 2 gap fix: 'Thanks, <Name>' / 'Regards, <Name>' sign-offs."""
    for text in [
        "Thanks, Ravi",
        "Regards, Priya",
        "Warm regards, Amit",
        "thanks and regards, Sunita",
        "Thanks & regards, Kunal",
        "Cheers, Neha",
        "Meeting moved to 5pm. Regards, Rohit",
    ]:
        clean, meta = scrub(text)
        assert "<NAME>" in clean, f"sign-off name not scrubbed: {text!r} -> {clean!r}"
        assert meta["n_names"] >= 1


def test_signoff_false_positives_left_alone():
    """Ordinary words after 'thanks/regards' must NOT be eaten."""
    for text in ["Thanks for your help", "thanks a lot for everything",
                 "regards this email is confidential"]:
        clean, _ = scrub(text)
        assert "<NAME>" not in clean, f"false sign-off scrub: {text!r} -> {clean!r}"


def test_signoff_stopwords_not_scrubbed():
    """'Thanks, Team' is not a person name."""
    clean, _ = scrub("Thanks, Team")
    assert "<NAME>" not in clean


def test_common_words_not_scrubbed_as_names():
    """'Dear Team' must NOT become 'Dear <NAME>'."""
    for text in ["Dear Team, hello", "This is urgent", "I am fine here"]:
        clean, _ = scrub(text)
        assert "<NAME>" not in clean, f"false name scrub: {text!r} -> {clean!r}"


# ---------------------------------------------------------------------------
# 8. THE BIG ONE: no original identifiers survive
# ---------------------------------------------------------------------------
MESSAGES_WITH_EVERYTHING = (
    "Dear Rohit, your A/C 123456789012 has Rs 5000. Your OTP is 482913. "
    "Send money to 98765 43210 or UPI id rohit@okhdfcbank, "
    "or mail rohit.k@example.com. Open http://192.168.0.5/kyc or bit.ly/ab12cd "
    "or sbi-kyc.xyz. I am Amit."
)


def test_no_original_identifiers_survive():
    """Nothing personal from the input may appear in the scrubbed output."""
    clean, meta = scrub(MESSAGES_WITH_EVERYTHING)

    # (a) every digit run of length >= 6 from the input must be gone
    original_digit_runs = {d for d in re.findall(r"\d{6,}", MESSAGES_WITH_EVERYTHING)}
    assert original_digit_runs, "test message should contain long digit runs"
    for run in original_digit_runs:
        assert run not in clean, f"digit run {run} leaked: {clean!r}"

    # (b) specific identifiers must be replaced by their tokens
    for token in ["<NAME>", "<ACCT>", "<OTP>", "<PHONE>", "<UPI>", "<EMAIL>", "<URL>"]:
        assert token in clean, f"expected token {token} missing: {clean!r}"

    # (c) fragments of the email / UPI / URL must be gone
    for fragment in ["rohit.k", "example.com", "okhdfcbank", "192.168", "bit.ly"]:
        assert fragment not in clean, f"fragment {fragment!r} leaked: {clean!r}"

    # (d) meta captured the structural signals we must keep
    assert meta["had_phone_number"] is True
    assert meta["url_type"] in {"ip_address", "shortener", "odd_tld", "official_looking"}
    assert meta["url_flags"], "url_flags should record every flag found"


def test_scrubbed_output_has_no_leftover_long_digits():
    """Any >=9 digit run after scrubbing is a bug (those must be <ACCT>)."""
    clean, _ = scrub(MESSAGES_WITH_EVERYTHING)
    assert not re.search(r"\d{9,}", clean), f"leftover long digits: {clean!r}"


def test_dotted_prose_is_not_eaten_as_url():
    """Hinglish chat with dots ('haan.. theek.hai') must survive untouched."""
    clean, meta = scrub("haan theek hai.. main aa raha hoon. koi baat nahi.")
    assert "theek" in clean and "haan" in clean
    assert "<URL>" not in clean
    assert meta["url_type"] == "none"


def test_scrub_is_safe_on_empty_and_none():
    assert scrub("")[0] == ""
    assert scrub(None)[0] == ""
