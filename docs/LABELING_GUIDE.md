# ScamLens Labeling Guide

How to label messages with `python -m src.label_cli`. Read `docs/TAXONOMY.md`
first — it defines every verdict and tag. This guide is the *procedure*.

---

## 1. Before you start

1. **Consent only.** Only label messages you received yourself, or that were
   given to you with permission (see `docs/DATA_ETHICS.md`).
2. **Skim the pool once.** Open `data/interim/unlabeled_pool.csv` and read
   through the scrubbed texts *before* labeling. The scrubber is imperfect —
   if you spot a leftover phone number or name, tell whoever ran the ingest,
   or fix the text manually in the CSV (replace it with `<PHONE>`/`<NAME>`).
3. Know your tag numbers. The tool prints them, but common ones are:
   1 `otp_or_pin_request`, 2 `urgency_threat`, 3 `bank_kyc_impersonation`,
   4 `authority_impersonation`, 5 `payment_or_upi_request`.

## 2. The labeling loop

For each message the tool asks, in order:

1. **verdict** — `s`cam / `u`nsure band / `f`afe, or `k` skip, `q`uit, `undo`.
   The model is **binary**: it trains and is scored on `scam` vs `safe` only.
   `suspicious` is the **"unsure band"**, not a class. Those rows are parked
   in `data/processed/suspicious_holdout.parquet` and Phase 4 uses them to
   test confidence-based abstention. Label them honestly anyway — they are
   valuable, just not a training label.
2. **tactics** — numbers like `1,3` then Enter; bare Enter = no tactics.
3. **confidence** — `h` / `m` / `l` (Enter = medium).
4. **date_received** — `YYYY-MM-DD`, `YYYY-MM`, or Enter if unknown.
   Use the date the message *arrived*, not a date quoted inside it.
5. **channel** — sms / whatsapp / email / call_transcript (Enter = unknown).

Your progress saves **after every message**. Quit whenever; resume later —
finished messages are skipped automatically. `undo` at the verdict prompt
removes the previous label.

## 3. Decision rules

- Read the whole message before deciding. One word ("OTP") is never enough.
- **Label the message, not the sender.** A genuine bank alert is `safe` even
  if you distrust banks.
- **Multi-tag freely.** "KYC expired, click <URL>, share OTP" is
  `bank_kyc_impersonation` + `suspicious_link` + `otp_or_pin_request`.
- **Suspicious is not a class — it is an unsure band.** Use it whenever the
  evidence does not reach "clearly fraud". Phase 4 turns low-confidence
  predictions into this band; it is never a training target in Phase 2/3.
- Confidence:
  - `high` — the verdict is obvious from the text alone (crystal-clear scam,
    obviously-genuine OTP delivery).
  - `medium` — you are fairly sure but there is one unresolved detail.
  - `low` — genuine guess: thin text, unknown sender, ambiguous link.

## 4. Borderline cases (quick reference)

| situation | verdict | confidence |
|---|---|---|
| bank alert, no link, official sender | safe | high |
| bank alert with `call <PHONE>` number | safe | high (`had_phone_number` records the signal) |
| OTP delivery message | safe | high |
| message that asks you to share an OTP | scam | high |
| OTP delivery + unknown short link | suspicious | low |
| prize you never entered, claim fee | scam | high |
| brand promo you don't remember subscribing to | safe (or suspicious if unknown sender) | medium/low |
| "Papa new number" from a known relative's thread | safe — **you know your own chat history** | high |
| "Papa new number" from an unknown number, asks for money | scam (`relative_in_trouble`) | high |
| collection call you actually owe | suspicious | low |
| completely unreadable / empty text | skip (`k`) and flag it | — |

## 5. Second labeler plan (Phase 2 prep)

To know whether our labels are *consistent* (not just one person's mood):

1. After the first pass, pick **50 messages at random** with seed 42
   (`numpy.random.default_rng(42)` over the labeled ids), stratified so ~25 are
   `scam`, ~15 `suspicious`, ~10 `safe` — including every hard case type.
2. A **second person labels them independently**. They read this guide and
   `docs/TAXONOMY.md`, but must NOT see your labels.
3. Export their file separately (they run `label_cli` with a different
   `labeling.default_labeler` and a separate `labeled.csv`).
4. In Phase 2 (`src/baselines.py`) we compute **Cohen's kappa** for the
   verdict (3-class) and per-tag agreement for tactics.
5. Target: kappa >= 0.6 before we trust the dataset. Below that, the taxonomy
   or this guide needs another revision round — do not "fix" it by re-labeling
   to match yourself.

## 6. Common mistakes to avoid

- Tagging `suspicious_link` on a message with **no link**.
- Marking every bank message `scam` because "banks never message like this".
- Forgetting that Hinglish phrasing ("kripya turant") is not itself a tactic —
  it usually pairs with `urgency_threat` or `payment_or_upi_request`.
- Setting date_received to *today* instead of the received date.
- Leaving confidence at the default `medium` for everything — it carries
  information only if you actually think about it.
