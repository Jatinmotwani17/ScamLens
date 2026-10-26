"""
src/label_cli.py - a tiny terminal tool for labeling messages one by one.

INPUT : data/interim/unlabeled_pool.csv  (made by src/ingest.py)
OUTPUT: data/interim/labeled.csv         (saved after EVERY message)

It is designed so you can quit at any time (Ctrl+C or 'q') and resume later -
already-labeled messages are skipped.

While labeling you can type:
    s   scam
    u   suspicious   (risky but not clearly fraud)
    f   safe         (legitimate message)
    k   skip this message for now (it stays unlabeled)
    q   quit and save
    u   (at the verdict prompt, after labeling) - not here; use 'undo' instead

At the tactic prompt: type numbers like  1,3,5  then Enter, or just Enter for
none. At every other prompt, Enter accepts the default shown in brackets.

Run:  python -m src.label_cli
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import CFG, CHANNELS, CONFIDENCE_LEVELS, SEED, TACTIC_TAGS, VERDICTS, data_path

# What each single-letter verdict key means.
VERDICT_KEYS = {"s": "scam", "u": "suspicious", "f": "safe"}


def load_existing_labels(path) -> pd.DataFrame:
    """Load labels we already saved, so labeling can resume after a quit."""
    if path.exists():
        df = pd.read_csv(path, dtype=str).fillna("")
        print(f"Resuming: {len(df)} message(s) already labeled in {path}")
        return df
    return pd.DataFrame(columns=[
        "id", "text", "verdict", "tactics", "channel", "script_style",
        "date_received", "source", "labeler", "label_confidence",
        "url_type", "had_phone_number", "notes",
    ])


def ask_verdict() -> str | None:
    """Ask for the verdict. Returns the verdict string, or None to quit."""
    while True:
        raw = input("  verdict [s=cam / u=suspicious / f=safe / k=skip / q=uit / undo]: ").strip().lower()
        if raw in VERDICT_KEYS:
            return VERDICT_KEYS[raw]
        if raw == "k":
            return "skip"
        if raw == "q":
            return None
        if raw == "undo":
            return "undo"
        print("  Please type s, u, f, k, q or undo.")


def ask_tactics() -> list[str]:
    """Show the numbered tactic list and return the chosen tags."""
    print("  tactics (enter numbers like 1,3,5 - Enter = none):")
    for i, tag in enumerate(TACTIC_TAGS, start=1):
        print(f"    {i:2}. {tag}")
    raw = input("  > ").strip()
    if not raw:
        return []
    picked: list[str] = []
    for part in raw.replace(" ", ",").split(","):
        if not part:
            continue
        if part.isdigit() and 1 <= int(part) <= len(TACTIC_TAGS):
            tag = TACTIC_TAGS[int(part) - 1]
            if tag not in picked:
                picked.append(tag)
        else:
            print(f"  (ignoring unknown number '{part}')")
    return picked


def ask_confidence() -> str:
    """h = high, m = medium, l = low. Default: medium."""
    while True:
        raw = input("  confidence [h/m/l, Enter=medium]: ").strip().lower()
        if raw == "":
            return "medium"
        if raw == "h":
            return "high"
        if raw == "m":
            return "medium"
        if raw == "l":
            return "low"
        print("  Please type h, m or l.")


def ask_date() -> str:
    """Optional date: YYYY-MM-DD or YYYY-MM. Empty = unknown."""
    while True:
        raw = input("  date_received [YYYY-MM-DD or YYYY-MM, Enter=unknown]: ").strip()
        if raw == "":
            return ""
        parts = raw.split("-")
        # Accept YYYY-MM or YYYY-MM-DD (digits only).
        ok_ym = (len(parts) == 2 and all(p.isdigit() for p in parts)
                 and len(parts[0]) == 4 and len(parts[1]) == 2)
        ok_ymd = (len(parts) == 3 and all(p.isdigit() for p in parts)
                  and len(parts[0]) == 4)
        if ok_ym or ok_ymd:
            return raw
        print("  Please use YYYY-MM-DD, YYYY-MM, or press Enter.")


def ask_channel() -> str:
    """Optional channel. Default: unknown."""
    print(f"  channel [{'/'.join(CHANNELS)}, Enter=unknown]:", end=" ")
    raw = input().strip().lower()
    if raw == "":
        return "unknown"
    if raw in CHANNELS:
        return raw
    print(f"  (unknown channel '{raw}', using 'unknown')")
    return "unknown"


def print_counts(labeled: pd.DataFrame) -> None:
    """Show the running verdict tally so label imbalance is visible early."""
    if labeled.empty:
        print("  (no labels yet)")
        return
    counts = labeled["verdict"].value_counts().to_dict()
    parts = [f"{v}: {counts.get(v, 0)}" for v in VERDICTS]
    print("  RUNNING TOTAL -> " + " | ".join(parts) + f" | total: {len(labeled)}")


def main() -> None:
    pool_path = data_path("unlabeled_pool")
    labeled_path = data_path("labeled")

    if not pool_path.exists():
        print(f"No pool file at {pool_path}.")
        print("Run 'python -m src.ingest' first.")
        return

    pool = pd.read_csv(pool_path, dtype=str).fillna("")
    labeled = load_existing_labels(labeled_path)
    done_ids = set(labeled["id"]) if not labeled.empty else set()

    labeler = CFG["labeling"]["default_labeler"]
    default_source = CFG["labeling"]["default_source"]

    todo = [row for _, row in pool.iterrows() if row["id"] not in done_ids]
    print(f"{len(todo)} message(s) left to label (of {len(pool)} in the pool).")
    print("Labels are saved after every message - you can quit anytime.")
    print("=" * 60)

    for row in todo:
        print(f"\n[{row['id']}]  script_style={row['script_style']}  "
              f"url_type={row['url_type']}  had_phone={row['had_phone_number']}")
        # Show the scrubbed message with visible line breaks.
        print("-" * 60)
        print("  " + row["text"].replace("\n", "\n  "))
        print("-" * 60)

        verdict = ask_verdict()
        if verdict == "undo":
            if labeled.empty:
                print("  Nothing to undo.")
            else:
                removed = labeled.iloc[-1]["id"]
                labeled = labeled.iloc[:-1]
                labeled.to_csv(labeled_path, index=False, encoding="utf-8")
                print(f"  Undid label for {removed}.")
            continue
        if verdict is None:  # quit
            break
        if verdict == "skip":
            print("  Skipped (will come back next run).")
            continue

        tactics = ask_tactics()
        confidence = ask_confidence()
        date_received = ask_date()
        channel = ask_channel()

        labeled = pd.concat([labeled, pd.DataFrame([{
            "id": row["id"],
            "text": row["text"],                    # already scrubbed text
            "verdict": verdict,
            "tactics": ";".join(tactics),           # semicolon-joined in CSV
            "channel": channel,
            "script_style": row["script_style"],
            "date_received": date_received,
            "source": row["source"] if row["source"] else default_source,
            "labeler": labeler,
            "label_confidence": confidence,
            "url_type": row["url_type"],
            "had_phone_number": str(row["had_phone_number"]).lower() == "true",
            "notes": "",
        }])], ignore_index=True)

        # Save after every single message so nothing is ever lost.
        labeled.to_csv(labeled_path, index=False, encoding="utf-8")
        print_counts(labeled)

    print("\nDone. Labels saved to", labeled_path)
    print_counts(labeled)


if __name__ == "__main__":
    print(f"(seed={SEED}) src/label_cli.py")
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C or Ctrl+D: everything already saved stays saved.
        print("\nInterrupted - all labels given so far are saved.")
        sys.exit(0)
