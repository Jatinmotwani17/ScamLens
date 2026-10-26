"""
src/tokenizer_inspect.py - which tokenizer handles Hinglish sensibly?

For each candidate model we take 200 TRAINING messages (stratified by
script_style, so roman Hinglish and Devanagari are not drowned out by English)
and report:
  * average tokens per message
  * average tokens per word
  * share of tokens that are a single character
  * how many messages would be truncated at max_length=128
  * the tokenization of 5 Hinglish training messages, side by side

A tokenizer that shreds Hinglish into single characters is a bad match: the
model spends its capacity re-learning words instead of scam patterns.

Only model ids that actually load from the HuggingFace Hub are used. If an id
fails, it is reported and skipped - never guessed.

Run:  python -m src.tokenizer_inspect
      python -m src.tokenizer_inspect --models google/muril-base-cased xlm-roberta-base
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.config import PROJECT_ROOT, SEED
from src.io_utils import load_messages

hf_logging.set_verbosity_error()

DEFAULT_MODELS = [
    "google/muril-base-cased",
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
    "l3cube-pune/hing-bert",          # optional; skipped cleanly if unavailable
]

# Used only if the training table has no roman-Hinglish rows at all.
FALLBACK_SAMPLES = [
    "Aapke account se Rs 4,999 debit hua hai. Agar aapne nahi kiya to turant call karein.",
    "Bhai kal shaam ko milte hain, ghar pe hi hoon main.",
    "KYC pending hai, aaj link par documents submit karo warna account band ho jayega.",
    "Parcel AWB 4821 out for delivery hai, delivery executive call karega.",
    "Papa ka naya number hai, urgent paise bhej do <UPI> par.",
]


def analyse(tokenizer, texts: list[str], max_length: int = 128) -> dict:
    """Token statistics for one tokenizer over a list of messages."""
    token_counts, word_counts, single, total_tokens, truncated = [], [], 0, 0, 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids)
        token_counts.append(len(toks))
        word_counts.append(max(len(str(text).split()), 1))
        single += sum(1 for t in toks if len(t.replace("▁", "")) <= 1)
        total_tokens += len(toks)
        truncated += int(len(toks) > max_length - 2)
    return {
        "avg_tokens_per_message": float(np.mean(token_counts)) if texts else 0.0,
        "avg_tokens_per_word": float(total_tokens / max(sum(word_counts), 1)),
        "single_char_share": float(single / max(total_tokens, 1)),
        "truncated_share": float(truncated / max(len(texts), 1)),
        "n_messages": len(texts),
        "n_tokens": total_tokens,
    }


def sample_by_style(df, rng, n: int = 200, min_per_style: int = 20) -> tuple[list[str], dict]:
    """Pick up to `n` training rows, giving every script style a voice."""
    styles = [s for s in ("roman_hinglish", "devanagari", "mixed", "english")
              if (df["script_style"] == s).any()]
    if not styles:
        return df["text"].astype(str).head(n).tolist(), {}
    per_style = max(min_per_style, n // len(styles))
    picked, counts = [], {}
    for style in styles:
        rows = df[df["script_style"] == style]
        take = min(per_style, len(rows))
        idx = rng.choice(len(rows), take, replace=False)
        picked.extend(rows.iloc[idx]["text"].astype(str).tolist())
        counts[style] = take
    # Trim/pad to exactly n where possible, then shuffle for an unbiased mix.
    rng.shuffle(picked)
    return picked[:n], counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare tokenizers on Hinglish")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args(argv)

    df = load_messages()
    train = df[(df["split"] == "train") & (df["source"] != "public_weak")]
    rng = np.random.default_rng(SEED)
    texts, style_counts = sample_by_style(train, rng, n=args.n)
    if not texts:
        raise SystemExit("No training rows - run python -m src.build_dataset first.")

    # Five actual roman-Hinglish training messages. They are synthetic/scrubbed,
    # so printing them here does not leak private data.
    hinglish = train[train["script_style"] == "roman_hinglish"]["text"].astype(str).tolist()
    samples = hinglish[:5] if len(hinglish) >= 5 else FALLBACK_SAMPLES
    if len(samples) < 5:
        samples = (samples + FALLBACK_SAMPLES)[:5]

    print(f"(seed={SEED}) src/tokenizer_inspect.py")
    print("=" * 60)
    print(f"{len(texts)} TRAINING messages, style mix: {style_counts}")

    lines = ["# ScamLens Phase 3 - tokenizer comparison", "",
             f"{len(texts)} random TRAINING messages, sampled per `script_style` "
             f"({style_counts}). Lower tokens/word and a lower single-character "
             "share is usually better for Hinglish.", "",
             "| model | avg tokens/message | avg tokens/word | single-char share | truncated at 128 | status |",
             "|---|---|---|---|---|---|"]
    loaded = {}
    for name in args.models:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            stats = analyse(tok, texts, max_length=args.max_length)
            by_style = {}
            for style in ("roman_hinglish", "devanagari", "mixed", "english"):
                sub = train[train["script_style"] == style]["text"].astype(str).tolist()
                if sub:
                    by_style[style] = analyse(tok, sub[:100], max_length=args.max_length)
            loaded[name] = (tok, stats, by_style)
            print(f"{name}: {stats['avg_tokens_per_word']:.2f} tokens/word, "
                  f"{stats['single_char_share']:.1%} single chars")
            lines.append(f"| `{name}` | {stats['avg_tokens_per_message']:.1f} "
                         f"| {stats['avg_tokens_per_word']:.2f} "
                         f"| {stats['single_char_share']:.1%} "
                         f"| {stats['truncated_share']:.1%} | loaded |")
        except Exception as e:
            print(f"{name}: SKIPPED ({type(e).__name__}: {str(e)[:120]})")
            lines.append(f"| `{name}` | - | - | - | - | skipped: {type(e).__name__} |")

    # ---- per-style detail -------------------------------------------------
    if loaded:
        lines += ["", "## By script style (up to 100 training rows per style)", "",
                  "| model | style | rows | tokens/message | tokens/word | single-char share |",
                  "|---|---|---|---|---|---|"]
        for name, (_, _, by_style) in loaded.items():
            for style, s in by_style.items():
                lines.append(f"| `{name}` | {style} | {s['n_messages']} "
                             f"| {s['avg_tokens_per_message']:.1f} "
                             f"| {s['avg_tokens_per_word']:.2f} "
                             f"| {s['single_char_share']:.1%} |")

    # ---- same sentences, side by side ------------------------------------
    lines += ["", "## Same 5 Hinglish messages, different tokenizers", "",
              "These are actual `roman_hinglish` TRAINING rows (synthetic/scrubbed).", ""]
    for idx, text in enumerate(samples, start=1):
        # PRIVACY: no message text in the report, only its size.
        lines += [f"**Message {idx}** ({len(str(text).split())} words, "
                  f"{len(str(text))} chars)", ""]
        for name, (tok, _, _) in loaded.items():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            toks = tok.convert_ids_to_tokens(ids)
            lines.append(f"* **`{name}`** ({len(toks)} tokens): {' | '.join(toks)}")
        lines.append("")

    # ---- plain-English read ----------------------------------------------
    if loaded:
        best = min(loaded.items(), key=lambda kv: kv[1][1]["avg_tokens_per_word"])
        lines += ["## Plain-English read", "",
                  f"Lowest token inflation: **`{best[0]}`** "
                  f"({best[1][1]['avg_tokens_per_word']:.2f} tokens per word). "
                  "A lower number means fewer pieces per Hinglish word, so more "
                  "of the 128-token budget is spent on meaning. Tokenizer "
                  "efficiency is only one factor: model pretraining data matters "
                  "at least as much, which is why E1/E3 compare the models "
                  "themselves.", ""]

    out = Path(PROJECT_ROOT) / "results" / "tokenizer_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
