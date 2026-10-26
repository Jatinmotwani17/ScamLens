# ScamLens Data Ethics

This project handles messages between human beings. These rules are not
optional.

## 1. Consent is required

- Use only messages **you received yourself**, or messages someone gave you
  **with explicit permission** ("you can use this SMS in my scam-detector
  project?").
- When collecting from family/friends: tell them what the project does, that
  their message will be scrubbed of personal details, and that they can ask
  for its removal at any time. If in doubt, don't collect.
- Never collect messages from a third party's phone without their knowledge.

## 2. What gets scrubbed, and where raw data lives

`src/scrub.py` replaces, before anything is stored outside `data/private/`:

| pattern | becomes |
|---|---|
| Indian phone numbers (all common formats) | `<PHONE>` |
| UPI ids (`name@bank`) | `<UPI>` |
| email addresses | `<EMAIL>` |
| 9+ digit runs (account/card numbers) | `<ACCT>` |
| OTP-looking 4–8 digit codes next to OTP/code/PIN | `<OTP>` |
| URLs (type preserved as `url_type`) | `<URL>` |
| name-like words after Dear/Mr/Shri/I am/... | `<NAME>` |

**The scrubber is imperfect.** After every ingest you must manually skim the
scrubbed pool (`data/interim/unlabeled_pool.csv`) for anything it missed.

Raw, unscrubbed text lives in **exactly one place**: `data/private/`, which is
gitignored. Nothing under `data/interim`, `data/processed` or `data/golden`
may contain raw personal information.

## 3. Never commit private data

- `.gitignore` excludes `data/private/`, `data/interim/`, and `*.parquet`.
- Before every commit, run `git status` and think about every file listed.
  If in doubt, don't add it.
- Never copy raw messages into notebooks, docstrings, commit messages, issue
  text, logs, or chat with third-party APIs (including LLMs).

## 4. Publishing results

- Published samples, papers, and demos may only use **scrubbed** text.
- Synthetic rows (`source="synthetic"`) may be published freely — they are
  placeholder text we generated.
- Public weak rows keep their original public license (e.g. CC BY 4.0 for the
  UCI SMS Spam Collection) — credit the source when you publish any of it.
- Do not publish the golden set in full; publish the schema and a few samples.

## 5. Never engage with scam senders

- Do **not** reply to, call, or "test" scammers to gather more data. It is
  dangerous and can compromise your safety and your accounts.
- Do not probe the URLs found in scam messages (no clicking, no visiting) —
  classification of `url_type` happens on the link *string* only.
- If a message contains a live threat toward you (real "digital arrest"
  style contact targeting your family), block and report it to the real
  authorities; do not keep the thread "for the dataset".

## 6. Retention and deletion

- If someone who contributed messages asks for removal: delete their rows from
  `data/private/raw_messages.txt`, re-run `ingest` → `label_cli` →
  `build_dataset` so the derived files regenerate without them.
- Keep `data/private/` backed up encrypted or not at all; it is disposable by
  design — the whole pipeline regenerates derived data from it.
