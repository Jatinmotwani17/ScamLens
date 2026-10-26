# ScamLens Label Taxonomy

This file is the **single source of truth** for what each label means. The
same vocabulary lives as constants in `src/config.py`. If you change one,
change the other.

A labeled message has:
- **one verdict** (scam / suspicious / safe), and
- **zero or more tactic tags** (the "why", only meaningful for scam/suspicious).

---

## 1. Verdicts

### `scam`
The message **clearly attempts to defraud** the recipient. At least one of:
- asks for money to be sent (UPI, gift card, crypto, cash),
- asks for credentials: OTP, PIN, card number, CVV, password, app screen share,
- impersonates an authority (bank, police, CBI, customs, telecom, court) **to
  extract money or information**,
- promotes a fake prize, fake job, or fake investment with a payment step.

If a reasonable Indian SMS/WhatsApp user would lose money or data by obeying
it, it is `scam`.

### `suspicious`
A **risky pattern that is not clearly fraud**, or **not enough information**:
- unknown sender + odd link you cannot verify,
- aggressive urgency but no clear ask,
- a real-looking brand message from an unofficial channel,
- commercial spam that is annoying but not stealing (this is also where weak
  public spam data lands).

Rule of thumb from `docs/LABELING_GUIDE.md`: **when unsure, choose
`suspicious` + confidence `low`.** Never guess `scam` on thin evidence.

### `safe`
A **legitimate** message — and note what "legitimate" includes:
- genuine bank debit/credit alerts (even with a "call us" phone number),
- real OTP **delivery** messages (the bank sending YOU a code),
- courier/delivery updates with tracking links,
- genuine promotions and marketing SMS,
- normal personal chat ("Papa new number hai" from an actual relative).

Delivering an OTP is safe; **asking** for one is scam. This distinction is the
most important one in the whole taxonomy.

---

## 2. Tactic tags (multi-label)

Tags explain **how** the scam works. Apply every tag that genuinely fits.
`safe` rows normally have no tags.

| tag | meaning |
|---|---|
| `otp_or_pin_request` | asks you to share an OTP, PIN, or "verification code" |
| `urgency_threat` | pressure or fear: "in 30 minutes", "final warning", legal action |
| `bank_kyc_impersonation` | pretends your bank KYC expired / account will freeze |
| `authority_impersonation` | police / CBI / customs / telecom / court; includes "digital arrest" |
| `payment_or_upi_request` | collect request, "send Rs 1 to verify", pay a fee to unlock |
| `suspicious_link` | a link you cannot trust (shortener, odd TLD, lookalike domain, raw IP) |
| `prize_or_lottery` | you "won" something; claim fee or details needed |
| `job_or_task_scam` | pay-to-join job, like-and-earn, data entry with registration fee |
| `investment_or_trading_scam` | guaranteed returns, double money, crypto/forex tips |
| `loan_app_harassment` | fake loan app: fee upfront, or threats to send your photos/contacts |
| `parcel_courier_scam` | fake parcel/customs/redelivery-charge messages |
| `utility_disconnection_scam` | electricity/gas/SIM will be disconnected; pay now |
| `relative_in_trouble` | "Papa new number hai", fake child/spouse needing urgent money |

---

## 3. Worked examples

All examples below are **obviously fictional placeholders** — angle-bracket
tokens stand in for anything real. They illustrate the *shape* of a tactic,
not real messages.

### `otp_or_pin_request`
> Scam: `<BANK> card blocked. Share the OTP sent to your number on <PHONE> within 30 minutes.`
> Safe contrast: `<BANK>: OTP <OTP> for your login. Do not share with anyone.` ← the bank *delivering* an OTP

### `urgency_threat`
> Scam: `FINAL WARNING: pay Rs. <AMOUNT> in 1 hour or your SIM is deactivated. <URL>`

### `bank_kyc_impersonation`
> Scam: `Your <BANK> KYC has expired. Update at <URL> or account will be frozen.`

### `authority_impersonation`
> Scam: `Cyber police: your parcel has illegal items. You are under digital arrest. Send all OTPs to <PHONE>.`

### `payment_or_upi_request`
> Scam: `To reactivate UPI, send Rs. 1 to <UPI>. Limit restores in 5 minutes.`

### `suspicious_link`
> Suspicious: `Your <APP> account had unusual login. Secure it: <URL>` (no other ask, link is odd)

### `prize_or_lottery`
> Scam: `CONGRATS! You won Rs. <AMOUNT> in the <BRAND> lucky draw. Claim: <URL>`

### `job_or_task_scam`
> Scam: `Part time job, earn Rs. <AMOUNT> daily at home. WhatsApp <PHONE> to start.`

### `investment_or_trading_scam`
> Scam: `Double money in 7 days. Send Rs. <AMOUNT> to <UPI>, get 2x back.`

### `loan_app_harassment`
> Scam: `Overdue 3 days. We will send goons. Pay Rs. <AMOUNT> to <UPI> in 2 hours.`

### `parcel_courier_scam`
> Scam: `Your parcel is held at customs. Pay Rs. <AMOUNT> at <URL> for release.`

### `utility_disconnection_scam`
> Scam: `Electricity disconnected tonight 8pm, unpaid Rs. <AMOUNT>. Pay now: <URL>`

### `relative_in_trouble`
> Scam: `Papa, naya number hai. Turant Rs. <AMOUNT> <UPI> par bhej do, zaroori hai.`

---

## 4. Hard cases (read this before labeling anything)

### A legit OTP vs a phishing OTP request
- **Safe**: `<APP> login OTP is <OTP>. Valid 10 min.` — an *incoming* code, no ask.
- **Scam**: `Your OTP is <OTP>. Reply with this code to verify.` — the message
  asking you to *reveal* a code (or a message that contains a code **and** asks
  you to forward/share it).
- Gray zone: a message delivers a code AND links to a page that asks for it →
  `scam` if the link is clearly fake, else `suspicious` + `low`.

### Promotional vs scam
- Real brand, real offer, no payment needed, has opt-out ("reply STOP") → `safe`.
- "You won a prize you never entered", asks for fee/details → `scam`.
- Offers from unknown senders with shortened links you cannot check →
  `suspicious` + `suspicious_link`.

### A bank alert that contains a link
- Genuine alerts from the bank's official sender ID with no link or a plain
  "call <PHONE>" → `safe` (record `had_phone_number=True`).
- Same wording but the link is a shortener / odd TLD / lookalike domain →
  `suspicious` at minimum; `scam` if it also asks for OTP/payment.
- When you cannot check the domain: `suspicious` + `low` confidence. The
  `url_type` column already preserves the structural signal for Phase 2.

### When unsure
1. Choose `suspicious` (not scam, not safe).
2. Set confidence to `low`.
3. Put your reasoning in `notes` ("link looks official but sender ID unknown").
4. Never inflate to `scam` "to be safe" — low-confidence scams poison training
   far more than honest `suspicious` labels do.
