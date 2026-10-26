"""
src/data_torch.py - PyTorch dataset + label maps for Phase 3.

WHAT THIS FILE DOES (in plain English):
  1. Turns the text labels into numbers:
       verdict "safe" -> 0, "scam" -> 1          (BINARY, no suspicious)
       tactics -> a 13-slot vector of 0/1        (multi-label)
     The mapping is saved as JSON so inference uses exactly the same numbers.
  2. Wraps the messages in a PyTorch Dataset.
  3. Provides a collate function that pads each BATCH to its own longest
     message (dynamic padding) instead of always padding to 128. This is
     faster and uses less memory.
  4. Computes class weights / pos_weight from the TRAIN split only, capped so
     one rare tag cannot explode the loss.

"suspicious" is not here on purpose: Phase 2.5 moved those rows to
data/processed/suspicious_holdout.parquet (Phase 4 abstention material).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import BINARY_VERDICTS, TACTIC_TAGS


# The fixed order matters: checkpoint + label map + predictions must agree.
VERDICT_CLASSES = list(BINARY_VERDICTS)          # ["safe", "scam"]


def build_label_maps() -> dict:
    """Return the JSON-serialisable label maps."""
    return {
        "verdict_classes": VERDICT_CLASSES,
        "verdict_to_id": {name: i for i, name in enumerate(VERDICT_CLASSES)},
        "id_to_verdict": {str(i): name for i, name in enumerate(VERDICT_CLASSES)},
        "tactic_tags": list(TACTIC_TAGS),
        "tactic_to_id": {tag: i for i, tag in enumerate(TACTIC_TAGS)},
        "id_to_tactic": {str(i): tag for i, tag in enumerate(TACTIC_TAGS)},
        "num_verdicts": len(VERDICT_CLASSES),
        "num_tags": len(TACTIC_TAGS),
    }


def save_label_maps(path: str | Path, maps: dict | None = None) -> Path:
    """Write the maps to JSON (called once per run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    maps = maps or build_label_maps()
    path.write_text(json.dumps(maps, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_label_maps(path: str | Path) -> dict:
    """Read the maps back. Raises a clear error if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"label maps not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def tactics_to_vector(tactics, maps: dict) -> np.ndarray:
    """Turn ['urgency_threat', ...] into a 13-dim 0/1 vector."""
    vec = np.zeros(maps["num_tags"], dtype=np.float32)
    if isinstance(tactics, (list, tuple, set, np.ndarray)):
        for tag in tactics:
            idx = maps["tactic_to_id"].get(str(tag))
            if idx is not None:
                vec[idx] = 1.0
    return vec


class VerdictTagDataset(Dataset):
    """One item = one message, tokenised (padding happens later in collate)."""

    def __init__(self, df: pd.DataFrame, tokenizer, maps: dict,
                 max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.maps = maps
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            truncation=True,
            max_length=self.max_length,
            padding=False,              # dynamic padding in collate_fn
            return_attention_mask=True,
        )
        verdict = self.maps["verdict_to_id"][str(row["verdict"])]
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "verdict": torch.tensor(verdict, dtype=torch.long),
            "tags": torch.tensor(tactics_to_vector(row["tactics"], self.maps),
                                 dtype=torch.float32),
        }


def make_collate_fn(pad_token_id: int):
    """Pad to the longest message IN THIS BATCH (not always max_length)."""
    def collate(batch: list[dict]) -> dict:
        lengths = [len(item["input_ids"]) for item in batch]
        max_len = max(lengths)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            n = len(item["input_ids"])
            input_ids[i, :n] = item["input_ids"]
            attention[i, :n] = item["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "verdict": torch.stack([item["verdict"] for item in batch]),
            "tags": torch.stack([item["tags"] for item in batch]),
        }
    return collate


def compute_class_weights(df: pd.DataFrame, maps: dict,
                          max_weight: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights from the TRAIN split only.

    * verdict weight = inverse frequency, so a rare class is not ignored.
    * tag pos_weight = (#negatives / #positives) per tag, capped. A tag with
      zero positives gets weight 1 and cannot be learned (logged by caller).
    """
    labels = df["verdict"].map(maps["verdict_to_id"]).to_numpy()
    counts = np.bincount(labels, minlength=maps["num_verdicts"]).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = np.clip(weights, 1.0, max_weight)
    verdict_weights = torch.tensor(weights, dtype=torch.float32)

    tag_matrix = np.stack([tactics_to_vector(t, maps) for t in df["tactics"]])
    pos = tag_matrix.sum(axis=0)
    neg = len(tag_matrix) - pos
    pos_weight = np.ones_like(pos, dtype=np.float32)
    nonzero = pos > 0
    pos_weight[nonzero] = np.minimum(neg[nonzero] / pos[nonzero], max_weight)
    return verdict_weights, torch.tensor(pos_weight, dtype=torch.float32)


def class_weight_summary(df: pd.DataFrame, maps: dict) -> dict:
    """Small human-readable summary for the training log."""
    labels = df["verdict"].map(maps["verdict_to_id"]).to_numpy()
    counts = np.bincount(labels, minlength=maps["num_verdicts"])
    tag_counts = {tag: int(sum(tag in t for t in df["tactics"]))
                  for tag in maps["tactic_tags"]}
    return {
        "verdict_counts": {name: int(counts[i])
                           for i, name in enumerate(maps["verdict_classes"])},
        "tag_positive_counts": tag_counts,
    }
