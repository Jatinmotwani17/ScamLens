"""
src/synth/generate.py - build the Phase 2 synthetic corpus (the real one).

This REPLACES the 63 Phase 1 seed rows (those live on as test fixtures in
tests/fixtures/synthetic_seed.csv). It composes messages from the Lego
pieces in src/synth/components.py:

  * generate_scams(n)       verdict=scam. A message is composed from ONE
                            template family (greeting + hook + demand +
                            closing), so the family name can be recorded in
                            template_family and later held out of training.
  * generate_suspicious(n)  verdict=suspicious. Ambiguous-but-risky messages:
                            unknown-sender promos with a link, unverifiable
                            loan offers, vague "call back" messages,
                            forwarded chain-style messages.
  * generate_safe(n)        verdict=safe. AT LEAST 50% HARD NEGATIVES -
                            legitimate-looking messages that deliberately
                            reuse scam vocabulary (bank alerts, genuine OTP
                            delivery, courier tracking, appointments). They
                            are marked notes="hard_negative". The rest are
                            normal personal chat / easy messages.
  * add_noise()             ~40% of ALL rows get realistic noise: typos,
                            dropped vowels, extra spaces, ALL CAPS, emoji,
                            Hinglish spelling variants, lookalike chars.

HARD RULES (asserted, not assumed):
  * every row: source="synthetic" - NEVER labelled as real
  * only placeholders <URL> / <PHONE> / <UPI> - no real links or numbers.
    We CRASH if any message ever contains a 7+ digit run.
  * template_family is always recorded (that is what the splitter needs).

Output: data/processed/synthetic.parquet
Run:    python -m src.synth.generate            (default sizes from config)
        python -m src.synth.generate --scam 100 --suspicious 30 --safe 100
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from src.config import CFG, MESSAGE_COLUMNS, SEED, TACTIC_TAGS, data_path
from src.ingest import detect_script_style
from src.scrub import scrub
from src.synth.components import (
    CLOSINGS,
    CONTEXTS,
    EASY_CHATTER,
    ENTITIES,
    GREETINGS,
    SAFE_CONTEXTS,
    SAFE_TAILS,
    TAILS,
    all_families,
    short_scam_families,
)

GEN_CFG = CFG["generate"]

# ---------------------------------------------------------------------------
# Style mix: how often each writing style is used for scam composition.
# ---------------------------------------------------------------------------
STYLE_WEIGHTS = {"english": 0.40, "roman_hinglish": 0.45, "devanagari": 0.15}

# Fake money amounts (never real account numbers - just prices/fees).
AMOUNTS = ["500", "250", "1,200", "2,999", "450", "12,500", "9,999", "75"]

EMOJIS = ["\U0001F525", "\U0001F4B0", "\U0001F600", "\U0001F4E6", "❗",
          "⚠️", "\u2764", "\U0001F1EE\U0001F1F3", "\U0001F911", "\u2757"]

# Hinglish spelling variants - realistic ways people type the same word.
HINGLISH_VARIANTS = {
    "karein": ["kariye", "karen"],
    "karega": ["karega", "karega"],
    "nahi": ["nahin", "nhi"],
    "hai": ["he", "h"],
    "karo": ["kro"],
    "bhejo": ["bhej"],
    "bhejein": ["bhejein", "bheje"],
    "karke": ["krke"],
    "padhe": ["padhi"],
}

# Lookalike characters scammers/typists produce.
LOOKALIKES = {"o": "0", "a": "@", "e": "3", "i": "!"}

VOWELS = "aeiou"

# Small, safe chat openers/closers. They add variety to ordinary personal
# messages (uniqueness fix) and never introduce a scam-like demand.
EASY_OPENERS = [
    "Hey", "Hi", "Namaste", "Good morning", "Good evening", "Arre", "Sun na",
    "Oye", "Hello", "Hi ji", "Namaste ji", "Kal", "Aaj", "Shaam ko",
    "Subah", "Weekend pe", "Ghar pe", "Office se", "Bhai", "Didi",
]
EASY_TAILS = [
    "Milte hain.", "Baat karte hain.", "Take care.", "Apna dhyan rakhna.",
    "See you.", "Jaldi milte hain.", "Theek hai?", "Chal bye.", "Okay.",
    "Haan bata dena.", "Koi jaldi nahi.", "Aaram se.", "Bas bata diya.",
    "Baad mein baat karenge.", "Thik hai.", "Chalo theek hai.",
    "Good night.", "Kal milte hain.", "Bye.", "Sab badhiya.",
]

# Prefixes used on legitimate service messages (adds variation safely).
SAFE_PREFIXES = [
    "Update:", "Notification:", "Service message:", "Reminder:",
    "Important update:", "Hello,", "Dear user,", "FYI:",
]

# Devanagari safe messages (Phase 2.5 style-balance fix). Without these,
# every devanagari row in the corpus is a scam and script_style alone is a
# perfect shortcut. These are ordinary, harmless messages.
DEVANAGARI_OPENERS = [
    "सुनो,", "अरे,", "नमस्ते,", "हाँ,", "ठीक है,", "अच्छा,",
    "कल,", "आज,", "शाम को,", "सुबह,",
]

DEVANAGARI_TAILS = [
    "मिलते हैं।", "ठीक है।", "अपना ध्यान रखना।", "बाद में बात करेंगे।",
    "जल्दी मिलते हैं।", "चलो ठीक है।", "शुभ रात्रि।", "बाय।",
    "कोई जल्दी नहीं।", "आराम से।", "फिर बात करते हैं।", "सब ठीक है।",
    "तुम भी ध्यान रखो।", "कल मिलते हैं।", "अच्छा, ठीक है।",
]

MIXED_SAFE = [
    "Kal milte hain, tab tak आराम करना।",
    "Mummy ne kaha khana garam है, jaldi aa jao.",
    "Office se nikal gaya, ghar पहुँचने वाला हूँ.",
    "Photos bhej dena, मैंने भी कुछ ली हैं.",
    "चाय ban gayi hai, aa jao.",
    "Doctor appointment कल है, time note kar lo.",
    "Rent transfer kar diya, receipt भेज देना.",
    "Bus late hai, बीस minute लगेंगे.",
    "Match शुरू हो गया, score kya है?",
    "Class कल 9 baje है, late mat hona.",
    "Parcel guard ke paas रखा है.",
    "Weekend pe ghar aa rahe ho? सब इंतज़ार कर रहे हैं.",
]

SAFE_DEVANAGARI = [
    "कल सुबह नौ बजे मिलते हैं।",
    "खाना बन गया है, आ जाओ।",
    "मैं घर पहुँच गया हूँ, चिंता मत करो।",
    "फोटो भेज देना जब समय मिले।",
    "आज ऑफिस से जल्दी निकलूंगा।",
    "डॉक्टर ने कहा सब ठीक है।",
    "कल की क्लास नौ बजे है।",
    "बाज़ार से दूध ले आना।",
    "मीटिंग के नोट्स भेज दो।",
    "वीकेंड पर घर आ रहे हो?",
    "बस लेट है, बीस मिनट लगेंगे।",
    "मैच देख रहे हो? स्कोर क्या है?",
    "चाय पीने चलते हैं शाम को।",
    "पार्सल गार्ड के पास रखा है।",
    "आंटी अब ठीक हैं।",
    "टिकट बुक हो गए हैं।",
    "किराया पांच तारीख को देना है।",
    "सोसाइटी की मीटिंग रविवार को है।",
    "स्कूल कल बारिश की वजह से बंद है।",
    "जन्मदिन की बहुत बहुत शुभकामनाएँ।",
    "मैं बस में हूँ, थोड़ी देर में पहुँचूंगा।",
    "आज खाने में क्या बनाओगे?",
    "कल के लिए मौसम ठीक है।",
    "नई किताब मिल गई, पढ़ने के बाद बताऊँगा।",
    "घर आकर आराम करना, बहुत काम था।",
    "शाम को बाज़ार जाना है क्या?",
    "मम्मी से बात हुई, सब ठीक है।",
    "कल सुबह योग की क्लास है।",
    "पानी की बोतल साथ ले जाना।",
    "दोस्त का फोन आया था, उसने हाल पूछा।",
    "मीटिंग कल दस बजे शिफ्ट हो गई है।",
    "बच्चों का स्कूल फीस जमा कर दी है।",
    "आज जल्दी सो जाना, कल सुबह निकलना है।",
    "गाड़ी की सर्विसिंग कल करवा लूंगा।",
    "तुम्हारा मैसेज मिल गया, धन्यवाद।",
]

# A digit run of 7+ digits in a generated message means a real-looking number
# slipped in. That is a HARD RULE violation - we crash loudly.
LONG_DIGIT_RUN = re.compile(r"\d{7,}")


# ---------------------------------------------------------------------------
# NOISE
# ---------------------------------------------------------------------------
def _noise_typo_swap(word: str, rng) -> str:
    """Swap two adjacent characters (common phone typo)."""
    if len(word) < 4:
        return word
    i = rng.integers(1, len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _noise_drop_vowel(word: str, rng) -> str:
    """Drop one vowel (people type 'msg' for 'msg' -> 'mssg' style)."""
    positions = [i for i, ch in enumerate(word) if ch.lower() in VOWELS]
    if not positions or len(word) <= 3:
        return word
    i = int(rng.choice(positions))
    return word[:i] + word[i + 1:]


def _noise_extra_space(word: str, rng) -> str:
    """Insert a space inside a word ('acc ount')."""
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    return word[:i] + " " + word[i:]


def _noise_lookalike(word: str, rng) -> str:
    """Replace one latin letter with a lookalike character."""
    for i, ch in enumerate(word):
        if ch.lower() in LOOKALIKES and rng.random() < 0.5:
            return word[:i] + LOOKALIKES[ch.lower()] + word[i + 1:]
    return word


def _apply_word_noise(word: str, rng) -> str:
    """Pick ONE random word-level noise for a single word."""
    op = rng.choice(["typo", "drop_vowel", "extra_space", "lookalike"])
    if op == "typo":
        return _noise_typo_swap(word, rng)
    if op == "drop_vowel":
        return _noise_drop_vowel(word, rng)
    if op == "extra_space":
        return _noise_extra_space(word, rng)
    return _noise_lookalike(word, rng)


def add_noise(text: str, rng) -> str:
    """Sprinkle realistic noise onto one message.

    Placeholder tokens (<URL>, <PHONE>, <UPI>) are NEVER touched - they are
    our safety guarantee and must survive verbatim.
    """
    ops = list(rng.choice(
        ["word_noise", "caps", "emoji", "hinglish", "double_space"],
        size=int(rng.integers(1, 3)),          # 1 or 2 different ops
        replace=False,
    ))
    words = text.split(" ")
    out: list[str] = []
    for w in words:
        if w.startswith("<") and w.endswith(">"):
            out.append(w)                       # protect placeholders
            continue
        if "word_noise" in ops and rng.random() < 0.30:
            # strip punctuation ends so we noise the word itself
            lead = w[:1] if w[:1] in "([,-" else ""
            core = w[len(lead):]
            out.append(lead + _apply_word_noise(core, rng))
        else:
            out.append(w)
    text = " ".join(out)

    if "caps" in ops and rng.random() < 0.6:
        text = text.upper()
    if "double_space" in ops:
        text = text.replace(" ", "  ", 1)
    if "emoji" in ops and rng.random() < 0.7:
        text = text + " " + str(rng.choice(EMOJIS))
    if "hinglish" in ops:
        for src, alts in HINGLISH_VARIANTS.items():
            if src in text.split(" ") and rng.random() < 0.5:
                text = text.replace(src, str(rng.choice(alts)), 1)
    return text


# ---------------------------------------------------------------------------
# COMPOSITION HELPERS
# ---------------------------------------------------------------------------
def _pick_style(rng) -> str:
    styles = list(STYLE_WEIGHTS)
    weights = list(STYLE_WEIGHTS.values())
    return str(rng.choice(styles, p=weights))


def _comp_text(component: dict, style: str) -> str:
    """Get a component's text in the wanted style, with a safe fallback."""
    text = component["text"].get(style)
    if not text:                                # devanagari can be "fewer"
        text = component["text"]["roman_hinglish"]
    return text


def _fill(text: str, entity: str, rng) -> str:
    """Substitute {ENTITY} and {AMOUNT} placeholders."""
    return text.replace("{ENTITY}", entity).replace("{AMOUNT}", str(rng.choice(AMOUNTS)))


def compose_scam(family: dict, rng) -> tuple[str, list[str]]:
    """Compose ONE scam message from a single template family.

    Returns (text, tactics). The tactic tags are the union of the family's
    own tags plus the tags of the hook and demand actually used.
    """
    style = _pick_style(rng)
    entity = str(rng.choice(ENTITIES[family["entity_group"]]))
    tags = set(family["tactics"])

    parts: list[str] = []
    if rng.random() < 0.70:                     # 30% of messages: no greeting
        parts.append(_fill(_comp_text(rng.choice(GREETINGS), style), entity, rng))
    # A neutral context line (75% of rows). It changes the wording, never the
    # ask or the tactics, so the scam stays exactly the same kind of scam.
    if rng.random() < 0.75:
        parts.append(_fill(_comp_text(rng.choice(CONTEXTS), style), entity, rng))
    hook = rng.choice(family["hooks"])
    parts.append(_fill(_comp_text(hook, style), entity, rng))
    demand = rng.choice(family["demands"])
    parts.append(_fill(_comp_text(demand, style), entity, rng))
    if rng.random() < 0.60:
        parts.append(_fill(_comp_text(rng.choice(TAILS), style), entity, rng))
    if rng.random() < 0.70:
        parts.append(_fill(_comp_text(rng.choice(CLOSINGS), style), entity, rng))
    tags.update(hook["tactics"])
    tags.update(demand["tactics"])
    return " ".join(p.strip() for p in parts if p.strip()), sorted(tags)


# ---------------------------------------------------------------------------
# SUSPICIOUS TEMPLATES - risky but not clearly fraud. Each entry is one
# family: (name, [template variants]). Templates may use {AMOUNT}/{ENTITY}.
# ---------------------------------------------------------------------------
SUSPICIOUS_FAMILIES = [
    ("sus_promo_f1", [
        "Get flat 80% off on all products. Shop now <URL>",
        "Mega sale! 80% off, sirf aaj ke liye. <URL> par jaayein",
        "Flash sale alert: up to 80% off. <URL>",
    ]),
    ("sus_promo_f2", [
        "Congratulations, you are selected for a cashback reward. Claim <URL>",
        "Aap cashback ke liye selected hain. Details <URL> par dekhein",
    ]),
    ("sus_promo_f3", [
        "Free gift hamper for you! Confirm address on <URL>",
        "Aapke liye free gift. Address confirm karein: <URL>",
    ]),
    ("sus_loan_f1", [
        "Instant personal loan up to Rs 50,000 with zero documents. Call <PHONE>",
        "Loan chahiye? Rs 50,000 tak turant. Call <PHONE>",
    ]),
    ("sus_loan_f2", [
        "Pre-approved credit line of Rs 75,000 unlocked. Check <URL>",
        "Credit limit Rs 75,000 ready. <URL> par check karein",
    ]),
    ("sus_loan_f3", [
        "Low interest loan offer, CIBIL score doesn't matter. WhatsApp <PHONE>",
        "CIBIL low hai? Loan phir bhi milega. WhatsApp <PHONE> par",
    ]),
    ("sus_callback_f1", [
        "Please call back on this number regarding an important matter.",
        "Zaroori baat ke liye is number par call back karein.",
    ]),
    ("sus_callback_f2", [
        "We tried reaching you. Call our helpdesk at your convenience.",
        "Aapse call karne ki koshish ki. Helpdesk par call karein.",
    ]),
    ("sus_callback_f3", [
        "Urgent: revert on the number given below at the earliest.",
        "Jald se jald neeche diye number par revert karein.",
    ]),
    ("sus_chain_f1", [
        "Forward this to 10 people within 1 hour and good luck will come. Ignore and bad luck in 2 days.",
        "Is message ko 10 logon ko bhejo, 1 ghante mein. Na bhejne par problem.",
    ]),
    ("sus_chain_f2", [
        "God message: send to 20 contacts and your wish will come true tonight.",
        "God message: 20 contacts ko bhejo, aaj raat wish poori hogi.",
    ]),
    ("sus_chain_f3", [
        "Video: minister caught on camera. Watch before it is deleted <URL>",
        "Video: mantri pakde gaye. Delete hone se pehle dekhein <URL>",
    ]),
    ("sus_mixed_f1", [
        "Your document is pending on our portal. Update at <URL>",
        "Aapka document portal par pending hai. <URL> par update karein.",
    ]),
    ("sus_mixed_f2", [
        "Profile viewing request for you. See who: <URL>",
        "Aapke liye profile viewing request. Dekhein: <URL>",
    ]),
    # Longer ambiguous messages - the length-leak fix needs the suspicious
    # class to overlap the scam length range too, not just safe.
    ("sus_long_f1", [
        "Your KYC documentation is incomplete as per our records. Please visit the link and submit the pending documents within 3 working days to avoid restriction of services on your account.",
        "Aapke account ke documents incomplete hain. Teen working days mein link par pending documents submit karein, warna services restrict ho sakti hain.",
    ]),
    ("sus_long_f2", [
        "We have been trying to reach you regarding an important update on your submitted application. Please contact our helpdesk between 10 am and 6 pm and quote your reference number.",
        "Aapki application ke important update ke liye hum aapse baar baar contact kar rahe hain. Helpdesk par 10 se 6 baje call karein.",
    ]),
    ("sus_long_f3", [
        "Congratulations! You have been shortlisted in our mega bumper draw. To know your prize and claim it, verify your details on the given link before the expiry of the offer.",
        "Badhai! Aap humare mega bumper draw mein shortlisted hue hain. Prize jaanne aur claim karne ke liye diye gaye link par details verify karein.",
    ]),
]

# ---------------------------------------------------------------------------
# SAFE TEMPLATES - normal messages. First the HARD NEGATIVES (they reuse
# scam vocabulary on purpose: bank words, OTP, links, payment words).
# notes="hard_negative" is set on every row from these families.
# ---------------------------------------------------------------------------
SAFE_HARD_FAMILIES = [
    ("safe_hard_bankdebit_f1", [
        "Rs {AMOUNT} debited from a/c XX1234 at ATM. Not you? Call <PHONE> to block.",
        "Rs {AMOUNT} debited from your account. Balance available. Helpline <PHONE>",
    ]),
    ("safe_hard_bankcredit_f1", [
        "Rs {AMOUNT} credited to a/c XX1234 on {DATE}. Avl Bal: as per passbook.",
        "Credit of Rs {AMOUNT} received in your account. Updated balance in app.",
    ]),
    ("safe_hard_neft_f1", [
        "INB txn of Rs {AMOUNT} success. Ref no. Keep for records. {ENTITY}",
        "NEFT transfer of Rs {AMOUNT} credited. A/c XX1234. {ENTITY}",
    ]),
    ("safe_hard_otp_f1", [
        "{OTP} is your OTP. Do NOT share it with anyone. {ENTITY} never asks for your OTP.",
        "{OTP} is your One Time Password, valid for 10 minutes. Do not share it.",
    ]),
    ("safe_hard_otp_f2", [
        "{OTP} OTP hai, kisi ko bhi share na karein. {ENTITY} kabhi OTP nahi maangta.",
        "Aapka login OTP hai {OTP}. Ise kisi ko mat batayein.",
    ]),
    ("safe_hard_courier_f1", [
        "Your consignment AWB {AWB} is out for delivery today. Contact <PHONE> for help.",
        "Parcel AWB {AWB} will be delivered between 10 am and 6 pm. Track on official app.",
    ]),
    ("safe_hard_appointment_f1", [
        "Appointment confirmed with Dr. on Thursday 11 am. Reach 15 minutes early. Carry your reports.",
        "Reminder: your appointment is tomorrow at 5 pm. Reply YES or NO to confirm.",
    ]),
    ("safe_hard_bill_f1", [
        "Your electricity bill of Rs {AMOUNT} for the month is due on the 18th. Pay via official app or bank.",
        "Water bill Rs {AMOUNT} due soon. Pay at the municipal office or official website.",
    ]),
    ("safe_hard_offer_f1", [
        "Festival offer: no-cost EMI on cards for 3 months. T&C on official website.",
        "You are eligible for a pre-approved upgrade on your card. Check your net banking.",
    ]),
    ("safe_hard_delivery_f1", [
        "Your order has been shipped and will arrive by Friday. Track in the app.",
        "Order delivered. Rate your experience in the app. Thank you for shopping with us.",
    ]),
]

# ---------------------------------------------------------------------------
# LONG SAFE MESSAGES (length-leak fix). The audit showed safe messages
# averaged ~58 chars vs ~120 for scams - a length-only model scored 0.64
# macro-F1. These multi-sentence LEGITIMATE messages (~90-170 chars) move
# the safe length distribution into the scam range. Most are hard negatives
# on purpose (bank/courier/bill vocabulary in a real message). Templates may
# use {AMOUNT}/{OTP}/{DATE}/{AWB}/{ENTITY} like the other safe families.
# ---------------------------------------------------------------------------
SAFE_LONG_FAMILIES = [
    ("safe_long_bankalert_f1", [
        "Rs {AMOUNT} debited from your a/c XX1234 on {DATE} towards UPI payee 'SHOPMART'. If you did not do this transaction, report it immediately in your banking app or call the helpline printed on the back of your card.",
        "Dear Customer, your a/c XX1234 has been debited by Rs {AMOUNT} on {DATE}. Available balance as per our records is updated in your passbook. In case of any discrepancy, please visit your nearest branch.",
    ]),
    ("safe_long_otp_f1", [
        "{OTP} is your one time password for transaction authentication. This OTP is valid for 10 minutes only. {ENTITY} will never ask you to share this OTP with anyone, including our staff.",
        "Aapka login OTP hai {OTP}. Yeh 10 minute ke liye valid hai. {ENTITY} kabhi bhi OTP share karne ko nahi kahega. Kisi ko bhi OTP na batayein.",
    ]),
    ("safe_long_courier_f1", [
        "Your parcel with AWB {AWB} has been out for delivery since 9:15 am and is expected between 2 pm and 5 pm today. The delivery executive will call you before arriving. Track live status in our official app.",
        "AWB {AWB}: Your shipment reached the local hub this morning and has been assigned for delivery. Please keep exact change ready if you have chosen cash on delivery.",
    ]),
    ("safe_long_bill_f1", [
        "Your electricity bill of Rs {AMOUNT} for the billing month is due on {DATE}. Late payment attracts a surcharge. Pay through the official app, net banking, or any authorised collection centre.",
        "Gas cylinder refill request received. Your cylinder will be delivered in the next 48 hours by the authorised distributor. Please keep Rs {AMOUNT} ready as per the subsidised price.",
    ]),
    ("safe_long_salary_f1", [
        "Salary credited for the month: Rs {AMOUNT} net of deductions. Please find the detailed payslip attached in the HR portal. Raise a ticket within 7 days for any query on the deduction shown.",
        "Your salary of Rs {AMOUNT} has been credited to your account. The breakup of allowances and deductions is available on the employee self-service portal under the payroll section.",
    ]),
    ("safe_long_chat_f1", [
        "Hey! Reaching the station around 7, can you pick me up from the main gate? Also bring the charger, mine died in the morning and I need the phone for the tickets.",
        "Haan bhai, kal shaam ko movie dekhne chalte hain. Booking main kar dunga, tu bas time se pahunch jaana. Popcorn wale zone mein milenge.",
    ]),
    ("safe_long_notice_f1", [
        "Dear parents, the annual sports day will be held on the school ground on Saturday. Students must report by 7:30 am in full uniform. Refreshments will be provided; no passes are required for entry.",
        "Notice: Water supply will remain suspended on Tuesday from 10 am to 4 pm for pipeline maintenance. Please store water accordingly. Inconvenience regretted. - Society Office",
    ]),
    ("safe_long_appointment_f1", [
        "Your appointment with the doctor on {DATE} at 11:00 am is confirmed. Please arrive 15 minutes early with your previous prescriptions and reports. Call the front desk to reschedule if needed.",
        "Reminder: your lab test slot is booked for {DATE} morning. Please visit the second floor counter and carry a photo ID. Results will be uploaded to your patient account within 24 hours.",
    ]),
]

# Easy negatives - ordinary chat and notices with NO scam vocabulary.
SAFE_EASY_FAMILIES = [
    ("safe_easy_chat_f1", [
        "Kahan ho? Dinner kar liya?",
        "Thik hoon, tu bata. Kal milte hain.",
        "Haan bhej diya, check kar lena.",
    ]),
    ("safe_easy_chat_f2", [
        "Meeting 5 baje hai, office hi rehna.",
        "Kal morning call karte hain. Good night!",
    ]),
    ("safe_easy_chat_f3", [
        "Send me the photos from Sunday when you are free.",
        "Thanks for the help today, really appreciated it.",
    ]),
    ("safe_easy_notice_f1", [
        "School closed tomorrow due to rain. - Principal",
        "Class test on Monday, chapters 4 to 7. All the best!",
    ]),
    ("safe_easy_notice_f2", [
        "Rent due on the 5th, please transfer as usual.",
        "Society meeting on Sunday at 11 am in the club house.",
    ]),
    ("safe_easy_delivery_f2", [
        "Bhai parcel aa gaya, ghar pe hi hoon.",
        "Kitaab bhej di hai, do din mein mil jayegi.",
    ]),
    ("safe_easy_chat_f4", [
        "Happy birthday! Have a great year ahead.",
        "Best wishes for the new house. Have a lovely day.",
    ]),
    ("safe_easy_reminder_f1", [
        "Reminder: call grandma in the evening.",
        "Gym 6 baje, late mat hona!",
    ]),
]


def _fill_safe(text: str, rng) -> str:
    """Fill placeholders used inside safe/suspicious templates."""
    otp = "".join(str(rng.integers(0, 10)) for _ in range(6))
    awb = "".join(str(rng.integers(0, 10)) for _ in range(4))
    date = str(rng.choice(["15th", "18th", "20th", "25th"]))
    entity = str(rng.choice(ENTITIES["banks"]))
    return (text.replace("{AMOUNT}", str(rng.choice(AMOUNTS)))
                .replace("{OTP}", otp)
                .replace("{AWB}", "AW" + awb + str(rng.integers(10, 99)))
                .replace("{DATE}", date)
                .replace("{ENTITY}", entity)
                .replace("{PHONE}", "<PHONE>"))


def generate_suspicious(n: int, rng) -> list[dict]:
    """Ambiguous messages: risky enough to flag, not clearly fraud.

    ~30% are the LONGER ambiguous ones (sus_long_*) so the suspicious class
    overlaps the scam length range (length-leak fix, see generate_scams).
    """
    rows = []
    long_fams = [f for f in SUSPICIOUS_FAMILIES if f[0].startswith("sus_long")]
    other_fams = [f for f in SUSPICIOUS_FAMILIES if not f[0].startswith("sus_long")]
    for i in range(n):
        if rng.random() < 0.30:
            fam_name, templates = long_fams[int(rng.integers(0, len(long_fams)))]
        else:
            fam_name, templates = other_fams[int(rng.integers(0, len(other_fams)))]
        parts = []
        # ALWAYS one context line (same meaning, more natural variety) and
        # often a tail. Always-on context stops short templates from producing
        # the same sentence hundreds of times (the Phase 2.5 uniqueness bug).
        parts.append(_comp_text(rng.choice(CONTEXTS), "english"))
        parts.append(_fill_safe(str(rng.choice(templates)), rng))
        # Tail is always present too: 24 x 18 = 432 safe wording variants per
        # template, which keeps the near-duplicate count low.
        parts.append(_comp_text(rng.choice(TAILS), "english"))
        text = " ".join(p.strip() for p in parts if p.strip())
        rows.append({
            "text": text,
            "verdict": "suspicious",
            "tactics": [],
            "template_family": fam_name,
            "notes": "",
        })
    return rows


def generate_safe(n: int, rng, min_hard_frac: float) -> list[dict]:
    """Safe messages, at least `min_hard_frac` of them HARD negatives."""
    n_hard = int(np.ceil(n * max(min_hard_frac, GEN_CFG["min_hard_negative_frac"])))
    n_easy = n - n_hard
    rows = []
    # ~30% of hard negatives are the LONG legitimate messages (length-leak
    # fix: safe text must overlap the scam length range).
    n_long = int(np.ceil(n_hard * 0.30))
    def hard_text(core: str) -> str:
        """Add harmless official-sounding context/tail around a safe message."""
        # Prefix + context are ALWAYS present. That gives every base template
        # 8 x 15 = 120 harmless variants, so repeated rows stay rare.
        parts = [str(rng.choice(SAFE_PREFIXES)),
                 _comp_text(rng.choice(SAFE_CONTEXTS), "english"),
                 core]
        if rng.random() < 0.60:
            parts.append(_comp_text(rng.choice(SAFE_TAILS), "english"))
        return " ".join(p.strip() for p in parts if p.strip())

    for i in range(n_long):
        fam_name, templates = SAFE_LONG_FAMILIES[int(rng.integers(0, len(SAFE_LONG_FAMILIES)))]
        text = hard_text(_fill_safe(str(rng.choice(templates)), rng))
        rows.append({
            "text": text,
            "verdict": "safe",
            "tactics": [],
            "template_family": fam_name,
            "notes": "hard_negative",
        })
    for i in range(n_hard - n_long):
        fam_name, templates = SAFE_HARD_FAMILIES[int(rng.integers(0, len(SAFE_HARD_FAMILIES)))]
        text = hard_text(_fill_safe(str(rng.choice(templates)), rng))
        rows.append({
            "text": text,
            "verdict": "safe",
            "tactics": [],
            "template_family": fam_name,
            "notes": "hard_negative",
        })
    for i in range(n_easy):
        # Three kinds of ordinary safe message: pure chatter (varied),
        # devanagari chat (script-balance fix), and an easy template with a
        # casual opener/tail. All three are harmless personal messages.
        roll = rng.random()
        if roll < 0.28:
            chat_idx = int(rng.integers(0, len(EASY_CHATTER)))
            # Several separate families (not one giant one): families are
            # indivisible during splitting, and one family would put every
            # roman-hinglish easy message into a single split.
            fam_name = f"safe_easy_chatter_f{(chat_idx % 12) + 1}"
            core = _comp_text(EASY_CHATTER[chat_idx], "roman_hinglish")
        elif roll < 0.50:
            dev_idx = int(rng.integers(0, len(SAFE_DEVANAGARI)))
            # 10 families so train/val/test_a/test_b each get devanagari SAFE
            # rows. With one family, script_style alone predicted the label.
            fam_name = f"safe_easy_devanagari_f{(dev_idx % 10) + 1}"
            core = str(SAFE_DEVANAGARI[dev_idx])
        elif roll < 0.55:
            mix_idx = int(rng.integers(0, len(MIXED_SAFE)))
            fam_name = f"safe_easy_mixed_f{(mix_idx % 4) + 1}"
            core = str(MIXED_SAFE[mix_idx])
        else:
            fam_name, templates = SAFE_EASY_FAMILIES[int(rng.integers(0, len(SAFE_EASY_FAMILIES)))]
            core = _fill_safe(str(rng.choice(templates)), rng)
            if rng.random() < 0.35:
                core += " " + _comp_text(rng.choice(EASY_CHATTER), "roman_hinglish")
        # Devanagari lines get Devanagari tails; Latin chat gets Latin tails.
        if fam_name.startswith("safe_easy_devanagari"):
            text = (f"{str(rng.choice(DEVANAGARI_OPENERS))} {core} "
                    f"{str(rng.choice(DEVANAGARI_TAILS))}")
        else:
            text = f"{str(rng.choice(EASY_OPENERS))}, {core} {str(rng.choice(EASY_TAILS))}"
        rows.append({
            "text": text,
            "verdict": "safe",
            "tactics": [],
            "template_family": fam_name,
            "notes": "",
        })
    rng.shuffle(rows)
    return rows


def generate_scams(n: int, rng) -> list[dict]:
    """Scam messages, one template family each.

    ~30% are TERSE one-liners (SCAM_SHORT_FAMILIES). The shortcut audit
    flagged msg_length as a shortcut (composed scams avg ~120 chars vs
    ~58 for safe), so short real-world-style scam SMS are mixed in to break
    the length signal.
    """
    families = all_families()
    short_fams = short_scam_families()
    short_rate = 0.30
    rows = []
    for i in range(n):
        if rng.random() < short_rate:
            fam = short_fams[int(rng.integers(0, len(short_fams)))]
            comp = fam["templates"][int(rng.integers(0, len(fam["templates"])))]
            style = _pick_style(rng)
            group = fam.get("entity_group", "none")   # short families may omit it
            entity = str(rng.choice(ENTITIES[group]))
            parts = []
            # Keep the CORE terse, but ALWAYS vary the surrounding wording so
            # short scams do not all collapse into the same few sentences.
            parts.append(_fill(_comp_text(rng.choice(CONTEXTS), style), entity, rng))
            parts.append(_fill(_comp_text(comp, style), entity, rng))
            if rng.random() < 0.45:
                parts.append(_fill(_comp_text(rng.choice(TAILS), style), entity, rng))
            text = " ".join(p.strip() for p in parts if p.strip())
            tactics = list(fam["tactics"]) + list(comp["tactics"])
            fam_name = fam["name"]
        else:
            family = families[int(rng.integers(0, len(families)))]
            text, tactics = compose_scam(family, rng)
            fam_name = family["name"]
        rows.append({
            "text": text,
            "verdict": "scam",
            "tactics": tactics,
            "template_family": fam_name,
            "notes": "",
        })
    return rows


def build_synthetic_frame(n_scam: int, n_suspicious: int, n_safe: int,
                          seed: int = SEED) -> pd.DataFrame:
    """Build the full synthetic table (scam + suspicious + safe + noise)."""
    rng = np.random.default_rng(seed)

    print(f"  generating {n_scam} scam rows ...")
    rows = generate_scams(n_scam, rng)
    print(f"  generating {n_suspicious} suspicious rows ...")
    rows += generate_suspicious(n_suspicious, rng)
    print(f"  generating {n_safe} safe rows (>=50% hard negatives) ...")
    rows += generate_safe(n_safe, rng, GEN_CFG["min_hard_negative_frac"])

    # ---- noise injection: ~40% of rows ------------------------------------
    noise_rate = float(GEN_CFG["noise_rate"])
    noisy_count = 0
    for r in rows:
        if rng.random() < noise_rate:
            r["text"] = add_noise(r["text"], rng)
            noisy_count += 1
            r["notes"] = (r["notes"] + ";noise").strip(";")

    # ---- scrub + finalise every row ---------------------------------------
    out = []
    for i, r in enumerate(rows, start=1):
        clean, meta = scrub(r["text"])
        had_phone = bool(meta["had_phone_number"]) or "<PHONE>" in r["text"]

        # HARD RULE: no real-looking phone numbers / long digit runs.
        if LONG_DIGIT_RUN.search(clean):
            raise AssertionError(
                f"HARD RULE VIOLATED: generated row {i} contains a 7+ digit run: {clean!r}"
            )

        out.append({
            "id": f"gen{i:05d}",
            "text": clean,
            "verdict": r["verdict"],
            "tactics": sorted(set(r["tactics"])),
            "channel": "",              # filled just below
            "script_style": detect_script_style(clean),
            "date_received": "",        # synthetic rows have no real date
            "source": "synthetic",      # ALWAYS synthetic. Never real.
            "template_family": r["template_family"],
            "labeler": "synth_gen_v1",
            "label_confidence": "high",  # by construction, not guessed
            "url_type": "other" if "<URL>" in clean else "none",
            "had_phone_number": had_phone,
            "template_cluster": -1,      # filled by dedup in build_dataset
            "split": "",                 # filled by splits2 in build_dataset
            "notes": r["notes"],
        })

    df = pd.DataFrame(out, columns=MESSAGE_COLUMNS)

    # Channels: scams mostly WhatsApp/SMS, safe mostly SMS/personal.
    chan_rng = np.random.default_rng(seed + 1)
    channels = []
    for verdict in df["verdict"]:
        if verdict == "scam":
            channels.append(str(chan_rng.choice(["whatsapp", "sms", "unknown"],
                                                p=[0.5, 0.45, 0.05])))
        else:
            channels.append(str(chan_rng.choice(["sms", "whatsapp", "email"],
                                                p=[0.6, 0.3, 0.1])))
    df["channel"] = channels

    # ---- summary ----------------------------------------------------------
    print(f"  noise applied to {noisy_count}/{len(df)} rows ({noisy_count / len(df):.0%})")
    hard = int((df["notes"].str.contains("hard_negative")).sum())
    n_safe_rows = int((df["verdict"] == "safe").sum())
    print(f"  hard negatives: {hard}/{n_safe_rows} safe rows "
          f"({hard / max(n_safe_rows, 1):.0%}, must be >= 50%)")
    # (skip the check when n_safe == 0 - debug runs may generate no safe rows)
    if n_safe_rows > 0:
        assert hard / n_safe_rows >= 0.50, "safe split must be >=50% hard negatives"

    covered = set()
    for tags in df.loc[df["verdict"] == "scam", "tactics"]:
        covered.update(tags)
    missing = [t for t in TACTIC_TAGS if t not in covered]
    if missing:
        print("  WARNING: tactic tags with no scam example:", missing)
    else:
        print(f"  all {len(TACTIC_TAGS)} tactic tags covered by scam rows.")

    return df


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 2 synthetic corpus.")
    parser.add_argument("--scam", type=int, default=int(GEN_CFG["n_scam"]))
    parser.add_argument("--suspicious", type=int, default=int(GEN_CFG["n_suspicious"]))
    parser.add_argument("--safe", type=int, default=int(GEN_CFG["n_safe"]))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    print(f"(seed={args.seed}) src/synth/generate.py")
    print("=" * 60)
    df = build_synthetic_frame(args.scam, args.suspicious, args.safe, seed=args.seed)

    out = data_path("synthetic_parquet")
    df.to_parquet(out, index=False)

    print("\nWROTE", out)
    print("  rows        :", len(df))
    print("  verdicts    :", df["verdict"].value_counts().to_dict())
    print("  styles      :", df["script_style"].value_counts().to_dict())
    print("  families    :", df["template_family"].nunique())
    print("  sources     :", df["source"].unique().tolist(), "(all synthetic by construction)")
    print("\nNext: python -m src.build_dataset   (cluster + split + validate)")


if __name__ == "__main__":
    main()
