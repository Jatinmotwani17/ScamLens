"""
src/io_utils.py - the ONE shared loader for the message table (Phase 2).

WHY THIS EXISTS
---------------
Parquet stores a list column (our `tactics`) as a numpy.ndarray when you read
it back, so after a round-trip `df["tactics"][0]` is an np.ndarray, NOT a
Python list. That difference bites later: isinstance(x, list) is False, JSON
dumping fails, and sets of tags behave differently. Every loader, baseline and
evaluation script therefore goes through load_messages(), which normalises
`tactics` into plain Python lists of strings exactly once.

Typical use:
    from src.io_utils import load_messages

    df = load_messages()                                  # messages.parquet
    df = load_messages("data/processed/synthetic.parquet")  # any other table

Run:  python -m src.io_utils    (prints a small self-check)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import data_path
from src.data_checks import as_tactics_list


def load_messages(path: str | Path | None = None) -> pd.DataFrame:
    """Read a messages parquet/CSV with `tactics` guaranteed to be list[str].

    path=None -> data/processed/messages.parquet (from configs/config.yaml).
    Raises a friendly FileNotFoundError if the table has not been built yet.
    """
    if path is None:
        p = data_path("messages_parquet")
    else:
        p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Build the table first (python -m src.build_dataset)."
        )

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p, dtype=str).fillna("")
    else:
        raise ValueError(f"load_messages: unsupported file type {p.suffix!r} ({p})")

    # ---- THE FIX: numpy array (or string, or anything) -> plain Python list --
    df = df.copy()
    df["tactics"] = [as_tactics_list(v) for v in df["tactics"]]

    # Missing/NaN template families simply mean "no family" -> "".
    if "template_family" in df.columns:
        df["template_family"] = (
            df["template_family"].fillna("").astype(str).str.strip()
        )

    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Tiny self-check you can run: python -m src.io_utils
    frame = load_messages()
    first = frame["tactics"].iloc[0]
    print(f"load_messages: {len(frame)} rows loaded from messages.parquet")
    print(f"  first tactics value : {first!r}  (type={type(first).__name__})")
    assert isinstance(first, list), "tactics must come back as a Python list"
    print("  OK - tactics round-trips as a real Python list.")
