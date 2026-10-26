"""
src/predict.py - run a saved Phase 3 model and keep EVERY prediction.

WHY KEEP EVERYTHING: Phase 4 needs the probabilities for calibration and for
the "suspicious = low confidence" abstention band. So each row keeps:
  id, text, true verdict/tags, predicted verdict/tags, all verdict
  probabilities, all 13 tag probabilities, and the real_stratum (if any).

Outputs:
  outputs/<run_name>/preds_val.parquet
  outputs/<run_name>/preds_test_a.parquet
  outputs/<run_name>/preds_test_b.parquet
  outputs/<run_name>/preds_test_real.parquet

Run:
  python -m src.predict --run_name e1_muril
  python -m src.predict --run_name e1_muril --splits test_real
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.config import SEED, data_path
from src.data_torch import (
    VerdictTagDataset,
    load_label_maps,
    make_collate_fn,
)
from src.io_utils import load_messages
from src.model import MultiTaskClassifier

hf_logging.set_verbosity_error()


@torch.no_grad()
def predict_frame(model, df, tokenizer, maps, max_length: int,
                  batch_size: int, device, cfg: dict | None = None) -> pd.DataFrame:
    """Return a DataFrame with one row per message and all probabilities."""
    if df.empty:
        return pd.DataFrame()
    ds = VerdictTagDataset(df, tokenizer, maps, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tokenizer.pad_token_id))
    verdict_probs, tag_probs = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        v_logits, t_logits = model(batch["input_ids"], batch["attention_mask"])
        verdict_probs.append(torch.softmax(v_logits, dim=-1).cpu().numpy())
        tag_probs.append(torch.sigmoid(t_logits).cpu().numpy())
    vp = np.concatenate(verdict_probs, axis=0)
    tp = np.concatenate(tag_probs, axis=0)

    out = df[["id", "text", "verdict", "tactics", "split", "source",
              "script_style", "channel", "template_family",
              "template_cluster", "notes"]].copy()
    if "real_stratum" in df.columns:
        out["real_stratum"] = df["real_stratum"].values
    out["true_tags"] = df["tactics"].apply(
        lambda t: [str(x) for x in t] if isinstance(t, (list, tuple, np.ndarray)) else [])
    out["pred_verdict"] = [maps["id_to_verdict"][str(i)] for i in vp.argmax(axis=1)]
    out["verdict_confidence"] = vp.max(axis=1)
    for i, name in enumerate(maps["verdict_classes"]):
        out[f"prob_{name}"] = vp[:, i]
    # `tags_trained` is False for the E2 ablation (lambda_tags=0). The columns
    # are still written (Phase 4 may inspect them), but nobody should score
    # them: that head never received a gradient.
    tags_trained = float(cfg.get("lambda_tags", 0) or 0) > 0
    out["tags_trained"] = tags_trained
    pred_tags = []
    for row in tp:
        pred_tags.append([maps["id_to_tactic"][str(i)] for i, v in enumerate(row)
                          if v >= 0.5] if tags_trained else [])
    out["pred_tags"] = pred_tags
    for i, tag in enumerate(maps["tactic_tags"]):
        out[f"prob_{tag}"] = tp[:, i]
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict with a saved Phase 3 run")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--splits", nargs="+",
                        default=["val", "test_a", "test_b", "test_real"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args(argv)

    # Resolve relative to the project root, exactly like src/train.py writes it.
    # Otherwise `python -m src.predict` from a subfolder would look in the wrong
    # place while src/evaluate.py scans PROJECT_ROOT/outputs.
    from src.config import PROJECT_ROOT

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = Path(PROJECT_ROOT) / output_root
    run_dir = output_root / args.run_name
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    maps = load_label_maps(run_dir / "label_maps.json")
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskClassifier(cfg["model_name"], maps["num_verdicts"],
                                maps["num_tags"])
    state = torch.load(run_dir / cfg.get("checkpoint_name", "best_model.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"(seed={SEED}) src/predict.py  run={args.run_name}  device={device}")

    df = load_messages()
    for split in args.splits:
        sub = df[df["split"] == split].reset_index(drop=True)
        if sub.empty:
            print(f"  {split}: EMPTY - skipped")
            continue
        pred = predict_frame(model, sub, tokenizer, maps,
                             int(cfg["max_length"]), args.batch_size, device, cfg)
        out = run_dir / f"preds_{split}.parquet"
        pred.to_parquet(out, index=False)
        print(f"  WROTE {out} ({len(pred)} rows)")
        if split == "test_real":
            print("  test_real kept real_stratum; report it PER STRATUM in evaluate.py")


if __name__ == "__main__":
    main()
