"""
src/make_real_test_template.py - collect REAL messages for test_real.

YOUR JOB (the only part no script can do):
  1. python -m src.make_real_test_template
     -> creates data/private/raw_messages.txt with instructions + a template.
  2. Paste REAL scam/legit messages you received (SMS/WhatsApp) into that
     file, one message per block, blocks separated by a line containing
     only three dashes: ---
     IMPORTANT: skim every message BEFORE saving. The scrubber removes
     phone numbers/URLs/emails, but a human must still check nothing
     sensitive is left.
  3. Run this command again with --ingest: it scrubs your messages into
     data/interim/unlabeled_pool.csv, opens the interactive labeler
     (src/label_cli.py, resumable - quit anytime with q), and saves labels
     into data/interim/labeled.csv.
  4. Run: python -m src.build_dataset
     Your labeled rows (source=self_collected/family_friends) are pinned to
     test_real automatically by src/splits2.py - they never train a model.

PRIVACY: data/private/ is gitignored (checked at startup, loudly).
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.config import SEED, data_path

TEMPLATE = """# ============================================================
# ScamLens - YOUR REAL MESSAGES (data/private/raw_messages.txt)
# This file is GITIGNORED. Raw private messages live ONLY here.
# ============================================================
#
# HOW TO FILL IT IN
#   * one message per block
#   * blocks separated by a line containing ONLY three dashes: ---
#   * delete these instruction lines if they annoy you
#   * SKIM every message yourself before keeping it - the scrubber
#     removes phone numbers / URLs / emails, but you are the last check
#
# Example block:
# ---
# <paste one real message here, e.g. an SMS you actually received>
# ---

---

"""
GITIGNORE_MARKER = "data/private/"


def ensure_private_dir() -> Path:
    """Create data/private/ and the template file if missing."""
    private = Path(data_path("private_raw")).parent
    private.mkdir(parents=True, exist_ok=True)

    # LOUD privacy check: data/private must be gitignored or the user could
    # accidentally commit raw personal messages.
    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    if gitignore.exists():
        text = gitignore.read_text(encoding="utf-8")
        if GITIGNORE_MARKER not in text:
            print("!" * 70)
            print(f"WARNING: {GITIGNORE_MARKER} is NOT in .gitignore!")
            print("Add that line BEFORE pasting private messages.")
            print("!" * 70)

    raw_path = Path(data_path("private_raw"))
    if not raw_path.exists():
        raw_path.write_text(TEMPLATE, encoding="utf-8")
        print(f"CREATED template: {raw_path}")
        print("Paste your real messages there (blocks separated by '---'),")
        print("then rerun:  python -m src.make_real_test_template --ingest")
    else:
        print(f"using existing {raw_path}")
    return raw_path


def run_labeling() -> None:
    """Ingest pasted messages, then open the interactive labeler."""
    raw_path = ensure_private_dir()
    blocks = [b.strip() for b in raw_path.read_text(encoding="utf-8").split("---")]
    real_blocks = [b for b in blocks
                   if b and not all(l.startswith("#") for l in b.splitlines())]
    if not real_blocks:
        print("No messages found in the template yet - paste some first!")
        return

    print(f"\n(step 1/2) ingesting {len(real_blocks)} pasted message block(s)...")
    from src.ingest import ingest
    ingest(raw_path=raw_path)

    print("\n(step 2/2) opening the labeler (resumable; press q to quit)...")
    from src.label_cli import main as label_main
    label_main()

    print("\nDONE. Next: python -m src.build_dataset")
    print("Your labeled rows are pinned to test_real automatically - they")
    print("never train a model; they only evaluate one.")


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Create/ingest the real-message collection template.")
    parser.add_argument("--ingest", action="store_true",
                        help="scrub pasted messages + run the labeler")
    args = parser.parse_args(argv)
    print(f"(seed={SEED}) src/make_real_test_template.py")
    print("=" * 60)
    if args.ingest:
        run_labeling()
    else:
        raw_path = ensure_private_dir()
        print(f"\nFill {raw_path} with real messages, then run with --ingest.")


if __name__ == "__main__":
    main()
