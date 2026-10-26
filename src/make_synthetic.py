"""
src/make_synthetic.py - ~60 CLEARLY-SYNTHETIC messages kept as TEST FIXTURES.

PURPOSE (and NON-purpose):
  * Purpose (Phase 2): these 63 Phase 1 seed rows are TEST FIXTURES only -
    they live in tests/fixtures/synthetic_seed.csv and are used by the test
    suite and by hand-checks. They are NOT training data anymore: the new
    generator in src/synth/generate.py produces the real synthetic corpus.
  * NON-purpose: these are NOT real data and must NEVER be used to evaluate a
    model. Every row has source="synthetic".

HONESTY RULES BUILT INTO THIS FILE:
  * Every generated row carries source="synthetic" and
    labeler="synthetic_seed_v1".
  * Text uses obvious placeholders only: <BANK>, <URL>, <PHONE>, <UPI>,
    <NAME>, <ACCOUNT>, <BRAND>, <AMOUNT>, <APP>. No real numbers, no real
    links, no real names.
  * Template text is written to be OBVIOUSLY generic placeholder text - it is
    a scaffold for the pipeline, not a replica of any real scam message.

Output: tests/fixtures/synthetic_seed.csv  (a test fixture, git-tracked)

Run:  python -m src.make_synthetic
"""
from __future__ import annotations

from src.config import SEED, TACTIC_TAGS, data_path
from src.ingest import detect_script_style
from src.scrub import scrub

SYN_COLUMNS = [
    "id", "text", "verdict", "tactics", "channel", "script_style",
    "date_received", "source", "labeler", "label_confidence",
    "url_type", "had_phone_number", "template_family", "notes",
]

# ---------------------------------------------------------------------------
# HARD NEGATIVES (25) - legitimate-looking messages that must NOT be flagged.
# These are the messages a naive "spam word" model gets wrong. All synthetic.
# ---------------------------------------------------------------------------
HARD_NEGATIVES = [
    # --- genuine banking alerts (English) ---
    ("safe", [], "sms", "Rs. <AMOUNT> debited from a/c <ACCOUNT> at <BANK> ATM. Bal: Rs. <AMOUNT>. Not you? Call <PHONE>."),
    ("safe", [], "sms", "Credit of Rs. <AMOUNT> in a/c <ACCOUNT> avail bal Rs. <AMOUNT>. -<BANK>"),
    ("safe", [], "sms", "A/c <ACCOUNT> debited by Rs. <AMOUNT> on date 01-01-2025 UPI Ref <ACCOUNT>. -<BANK>"),
    # --- genuine banking alerts (Hinglish) ---
    ("safe", [], "sms", "<BANK>: a/c <ACCOUNT> se Rs. <AMOUNT> kat gaye hai. Agar aapne nahi kiya toh <PHONE> par call karein."),
    ("safe", [], "sms", "Priya: Rs. <AMOUNT> credit hua hai a/c <ACCOUNT> mein. Avail bal Rs. <AMOUNT>. -<BANK>"),
    # --- genuine OTP delivery (the OTP itself is fine; asking for it is not) ---
    ("safe", [], "sms", "<APP> ka login OTP hai <OTP>. Ise kisi ko mat batayein. Valid 10 minute."),
    ("safe", [], "sms", "<OTP> is your one time password for <BANK>. Do not share with anyone."),
    ("safe", [], "sms", "<BANK> transaction OTP: <OTP>. Valid for this transaction only."),
    # --- courier / delivery updates ---
    ("safe", [], "sms", "Your parcel <ACCOUNT> is out for delivery today. Delivery executive will call <PHONE>."),
    ("safe", [], "sms", "Aapka parcel aaj deliver hoga. Tracking id <ACCOUNT>. Delivery partner: <BRAND>."),
    ("safe", [], "sms", "<BRAND>: Order <ACCOUNT> shipped, arriving by 01-01-2025. Track: <URL>"),
    # --- appointment / reminder messages ---
    ("safe", [], "sms", "Reminder: appointment with Dr. <NAME> on 01-01-2025 at 10:00 am. Reach 15 min early. -<BRAND> Hospital"),
    ("safe", [], "sms", "Aapki appointment 01-01-2025 ko <BRAND> Clinic mein hai subah 10 baje. -<BRAND>"),
    ("safe", [], "sms", "Class reunion on 01-01-2025 at 6pm, <BRAND> Cafe. Reply YES if coming."),
    # --- genuine offers / transactional promos ---
    ("safe", [], "sms", "<BRAND> festive sale: flat 20% off till 01-01-2025. Use code SYNTH20. T&C apply. Opt-out: reply STOP"),
    ("safe", [], "sms", "Aapke <BRAND> wallet mein 2 coupons hai jo 01-01-2025 ko expire ho rahe hai. Open <APP> to use."),
    ("safe", [], "sms", "Your movie booking for SYNTH CITY is confirmed. Seats: A1, A2. Show 01-01-2025 7pm."),
    # --- salary / bills / finance (legit) ---
    ("safe", [], "sms", "Salary of Rs. <AMOUNT> credited to a/c <ACCOUNT> for the month of JAN 2025. -<BANK>"),
    ("safe", [], "sms", "<BANK>: Your credit card bill of Rs. <AMOUNT> is due on 01-01-2025. Pay via official <APP>."),
    ("safe", [], "sms", "Your electricity bill of Rs. <AMOUNT> for consumer <ACCOUNT> is due. Pay before 01-01-2025. -<BRAND>"),
    ("safe", [], "sms", "EMI of Rs. <AMOUNT> debited from a/c <ACCOUNT> on 01-01-2025. Loan a/c <ACCOUNT>. -<BANK>"),
    # --- normal personal chat (Hinglish) ---
    ("safe", [], "whatsapp", "Bhai, main station pahunch gaya hoon. Tu aa jaa, main <PHONE> par call karta hoon."),
    ("safe", [], "whatsapp", "Papa, mujhe ek help chahiye thi... bas ye message tha, koi zaroori nahi. Dinner mein baat karte hai."),
    ("safe", [], "email", "Hi Team, attached is the meeting notes from today. Thanks, <NAME>"),
    ("safe", [], "whatsapp", "Happy birthday beta! Bahut khush raho. -Dadi"),
]

# ---------------------------------------------------------------------------
# SCAM TEMPLATES (35) - every tactic tag covered at least twice.
# verdict is 'scam' unless the message is genuinely ambiguous ('suspicious').
# ---------------------------------------------------------------------------
# Format: (verdict, tactics, channel, text)
SCAM_TEMPLATES = [
    # --- otp_or_pin_request (3) ---
    ("scam", ["otp_or_pin_request"], "sms",
     "<BANK> blocked your card. To unblock share the OTP sent to your number on <PHONE>. Urgent: within 30 min or card closes."),
    ("scam", ["otp_or_pin_request", "urgency_threat"], "whatsapp",
     "Your <APP> account will be suspended. Send the 6 digit OTP you just received to <PHONE> immediately to verify."),
    ("scam", ["otp_or_pin_request", "authority_impersonation"], "sms",
     "CBI cyber cell: your KYC is incomplete. Dial <PHONE> and tell them the OTP you receive to avoid account freeze."),
    # --- urgency_threat (3) ---
    ("scam", ["urgency_threat"], "sms",
     "FINAL WARNING: your <BANK> account will be BLOCKED in 2 hours. Update details instantly at <URL>"),
    ("scam", ["urgency_threat"], "sms",
     "Last notice: pay Rs. <AMOUNT> fine within 1 hour or your SIM will be deactivated. Recharge now: <URL>"),
    ("scam", ["urgency_threat", "utility_disconnection_scam"], "whatsapp",
     "Electricity will be disconnected TONIGHT 8pm due to unpaid Rs. <AMOUNT>. Pay instantly: <URL>"),
    # --- bank_kyc_impersonation (3) ---
    ("scam", ["bank_kyc_impersonation"], "sms",
     "<BANK>: Your KYC has expired. Re-KYC now at <URL> or your account will be frozen."),
    ("scam", ["bank_kyc_impersonation", "suspicious_link"], "sms",
     "Dear customer, update your <BANK> KYC before 01-01-2025: <URL>. Failure will deactivate a/c <ACCOUNT>."),
    ("scam", ["bank_kyc_impersonation", "otp_or_pin_request"], "whatsapp",
     "Namaste, <BANK> se bol rahe hai. KYC update ke liye OTP bhej dijiye <PHONE> par, warna account band ho jayega."),
    # --- authority_impersonation / digital arrest (3) ---
    ("scam", ["authority_impersonation", "urgency_threat"], "whatsapp",
     "This is the police cyber unit. Your parcel contains illegal items. You are under digital arrest. Do not disconnect. Send all OTPs to <PHONE> for 'verification'."),
    ("scam", ["authority_impersonation"], "sms",
     "Customs dept: parcel <ACCOUNT> in your name has banned notes. Pay Rs. <AMOUNT> clearance fee to <UPI> immediately or officers will reach your home."),
    ("scam", ["authority_impersonation", "payment_or_upi_request"], "whatsapp",
     "Telecom regulatory office: your SIM will be closed as it sent a fake message. To keep it, deposit Rs. <AMOUNT> via UPI to <UPI> now."),
    # --- payment_or_upi_request (3) ---
    ("scam", ["payment_or_upi_request"], "sms",
     "To reactivate your UPI, send Rs. 1 to <UPI>. Your limit will be restored in 5 minutes."),
    ("scam", ["payment_or_upi_request"], "whatsapp",
     "Bhai, naya UPI banaya hai. Ek rupya bhej do <UPI> par, uske baad main tujhe 500 wapas bhej dunga."),
    ("scam", ["payment_or_upi_request", "urgency_threat"], "sms",
     "Rs. <AMOUNT> pending on your name. Pay via UPI <UPI> in 1 hour to avoid credit score hit."),
    # --- suspicious_link (3) ---
    ("scam", ["suspicious_link"], "sms",
     "<APP>: Your account has unusual login. Secure it now: <URL>"),
    ("scam", ["suspicious_link"], "sms",
     "Aapka courier ruka hai. Address confirm karne ke liye yahan click karein: <URL>"),
    ("scam", ["suspicious_link", "prize_or_lottery"], "sms",
     "CONGRATS! You won Rs. <AMOUNT> in the <BRAND> lucky draw. Claim before 01-01-2025: <URL>"),
    # --- prize_or_lottery (2) ---
    ("scam", ["prize_or_lottery"], "sms",
     "Lucky draw result: your mobile number won Rs. <AMOUNT>. Share your bank details at <URL> to receive the prize."),
    ("scam", ["prize_or_lottery", "payment_or_upi_request"], "whatsapp",
     "Aapka number lottery mein select hua hai. Prize lene ke liye Rs. <AMOUNT> registration fee <UPI> par bhejein."),
    # --- job_or_task_scam (3) ---
    ("scam", ["job_or_task_scam"], "whatsapp",
     "Part time job: earn Rs. <AMOUNT> daily sitting at home. Just like and share 5 videos. Message <PHONE> on WhatsApp to start."),
    ("scam", ["job_or_task_scam", "payment_or_upi_request"], "whatsapp",
     "Data entry job opening. Registration fee Rs. <AMOUNT> payable to <UPI>. Daily payout after 3 tasks."),
    ("scam", ["job_or_task_scam", "investment_or_trading_scam"], "sms",
     "Daily income plan: invest Rs. <AMOUNT>, earn Rs. <AMOUNT> every day for 30 days. Join now: <URL>"),
    # --- investment_or_trading_scam (3) ---
    ("scam", ["investment_or_trading_scam"], "whatsapp",
     "Sir, our trading group gave 40% profit last month. Add funds to <ACCOUNT> and we will trade for you. Guaranteed return."),
    ("scam", ["investment_or_trading_scam"], "sms",
     "Double your money in 7 days! Send USDT worth Rs. <AMOUNT> to <UPI> and receive 2x back. Limited slots: <URL>"),
    ("scam", ["investment_or_trading_scam", "urgency_threat"], "whatsapp",
     "Last 2 slots left in our crypto plan. Miss it and you lose Rs. <AMOUNT> profit. Deposit today to <ACCOUNT>."),
    # --- loan_app_harassment (2) ---
    ("scam", ["loan_app_harassment", "urgency_threat"], "sms",
     "Your instant loan of Rs. <AMOUNT> is approved. Pay processing fee Rs. <AMOUNT> NOW to <UPI> or the amount will be sent to your contacts."),
    ("scam", ["loan_app_harassment"], "whatsapp",
     "Overdue since 3 days. We are sending goons to your address. Pay Rs. <AMOUNT> to <UPI> within 2 hours. -<BRAND> Loans Recovery"),
    # --- parcel_courier_scam (3) ---
    ("scam", ["parcel_courier_scam", "authority_impersonation"], "sms",
     "India Post: parcel <ACCOUNT> held at customs. Pay Rs. <AMOUNT> at <URL> or parcel will be returned."),
    ("scam", ["parcel_courier_scam"], "sms",
     "Your parcel could not be delivered. Update address and pay Rs. 1 re-delivery charge: <URL>"),
    ("scam", ["parcel_courier_scam", "payment_or_upi_request"], "whatsapp",
     "Courier company: aapka parcel fas gaya hai. Rs. <AMOUNT> shipping charge <UPI> par bhejein, warna parcel destroy ho jayega."),
    # --- utility_disconnection_scam (2) ---
    ("scam", ["utility_disconnection_scam"], "sms",
     "<BRAND> Gas: your cylinder delivery is on hold. Pay Rs. <AMOUNT> at <URL> to resume connection."),
    ("scam", ["utility_disconnection_scam", "payment_or_upi_request"], "sms",
     "SIM bnd hogi 2 ghante mein. Rs. <AMOUNT> recharge karein UPI <UPI> par ya number deactivate ho jayega."),
    # --- relative_in_trouble (3) ---
    ("scam", ["relative_in_trouble"], "whatsapp",
     "Papa, naya number hai mera. Ek important payment karni hai, Rs. <AMOUNT> is number par bhej do <UPI>. Baad mein baat karta hoon."),
    ("scam", ["relative_in_trouble", "urgency_threat"], "whatsapp",
     "Beta, main hospital mein hoon. Turant Rs. <AMOUNT> <UPI> par bhej do, operation kal subah hai."),
    ("scam", ["relative_in_trouble", "payment_or_upi_request"], "whatsapp",
     "Bhai mai hu. Phone toot gaya. Ye mera naya number. 2000 rupay is UPI pe bhej do <UPI>, ghar walo ko mat batana."),
    # --- suspicious verdict: risky but NOT clearly fraud (2) ---
    ("suspicious", ["suspicious_link"], "sms",
     "Your <BRAND> rewards of Rs. <AMOUNT> expire soon. Redeem here: <URL> Reply STOP to opt out."),
    ("suspicious", [], "sms",
     "Unknown sender: we need a small verification from your side. Call back on <PHONE> during office hours."),
]


def build_synthetic() -> list[dict]:
    """Combine hard negatives and scam templates into schema-ready rows."""
    rows: list[dict] = []
    counter = 0

    def add(verdict: str, tactics: list[str], channel: str, text: str) -> None:
        nonlocal counter
        counter += 1
        clean, meta = scrub(text)  # scrub for uniformity (placeholders survive)
        had_phone = bool(meta["had_phone_number"]) or "<PHONE>" in text
        rows.append({
            "id": f"syn{counter:04d}",
            "text": clean,
            "verdict": verdict,
            "tactics": tactics,                       # real list, not a string
            "channel": channel,
            "script_style": detect_script_style(clean),
            "date_received": "",                      # unknown by construction
            "source": "synthetic",                    # ALWAYS synthetic
            "labeler": "synthetic_seed_v1",
            "label_confidence": "high",               # by construction, not guessed
            # Synthetic text uses <URL> placeholders, so the real URL type is
            # whatever the template declared by having a link at all:
            "url_type": "other" if "<URL>" in text else "none",
            "had_phone_number": had_phone,
            "template_family": "",   # Phase 1 seeds have no family (fixtures)
            "notes": "SYNTHETIC placeholder text - never use for evaluation",
        })

    for verdict, tactics, channel, text in HARD_NEGATIVES:
        add(verdict, tactics, channel, text)
    for verdict, tactics, channel, text in SCAM_TEMPLATES:
        add(verdict, tactics, channel, text)

    return rows


def main() -> None:
    print(f"(seed={SEED}) src/make_synthetic.py")
    print("=" * 60)
    rows = build_synthetic()

    import pandas as pd

    df = pd.DataFrame(rows, columns=SYN_COLUMNS)
    out_path = data_path("synthetic_seed")
    df.to_csv(out_path, index=False, encoding="utf-8")

    n_syn = int((df["source"] == "synthetic").sum())
    print(f"Generated {len(df)} synthetic rows ({n_syn} with source='synthetic').")
    print("  verdict counts:", df["verdict"].value_counts().to_dict())
    print("  channel counts:", df["channel"].value_counts().to_dict())
    print("  script_style  :", df["script_style"].value_counts().to_dict())

    # Confirm every tactic tag in the taxonomy is demonstrated at least once.
    covered = set()
    for t in df["tactics"]:
        covered.update(t if isinstance(t, list) else [])
    missing = [t for t in TACTIC_TAGS if t not in covered]
    if missing:
        print("  WARNING: tactic tags with NO synthetic example:", missing)
    else:
        print("  all", len(TACTIC_TAGS), "tactic tags have >=1 synthetic example.")

    print("Wrote", out_path)
    print("REMINDER: these Phase 1 seed rows are TEST FIXTURES ONLY.")
    print("Training data comes from src/synth/generate.py (Phase 2 generator).")


if __name__ == "__main__":
    main()
