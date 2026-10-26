"""
src/e4_cv.py - E4: cluster-grouped cross-validation on REAL messages.

This is the ONLY experiment where real messages enter training. It answers:
"if I had labelled real messages, how well would the model do on a NEW one?"

THE GATE (loudly enforced):
  * at least 50 in-domain real messages (user_real_scam + user_real_safe)
  * at least 20 user_real_safe hard negatives
If the gate fails, E4 is SKIPPED and nothing is trained. This prevents a
5-row "cross-validation" from being reported as a result.

Folds are grouped by `template_cluster`, so two near-duplicate messages can
never sit in different folds (that would leak the answer).

Run: python -m src.e4_cv --config configs/train.yaml --model google/muril-base-cased
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT_ROOT, SEED


def gate_ok(real: pd.DataFrame) -> tuple[bool, int, int]:
    """The Phase 3 amendment gate, exactly as written.

    <50 in-domain real messages OR <20 user_real_safe hard negatives -> skip.
    """
    n_scam = int((real["real_stratum"] == "user_real_scam").sum())
    n_safe = int((real["real_stratum"] == "user_real_safe").sum())
    return (n_scam + n_safe >= 50 and n_safe >= 20), n_scam + n_safe, n_safe


def load_frames() -> dict[str, pd.DataFrame]:
    """Same split loading as src.train.load_frames, without importing torch.

    Duplicated on purpose: E4's gate must be runnable on a machine where
    torch/transformers are not installed, so it can fail loudly instead of
    crashing with an ImportError before it ever sees the data.
    """
    from src.io_utils import load_messages

    df = load_messages()
    if "suspicious" in set(df["verdict"]):
        raise SystemExit("suspicious rows in the main table - run src.build_dataset")
    return {
        "train": df[(df["split"] == "train") & (df["source"] != "public_weak")].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test_a": df[df["split"] == "test_a"].reset_index(drop=True),
        "test_b": df[df["split"] == "test_b"].reset_index(drop=True),
        "test_real": df[df["split"] == "test_real"].reset_index(drop=True),
    }


def run_fold(train_df, test_df, cfg, maps, tokenizer, device, fold: int) -> dict:
    """Train one fold and score the held-out real messages.

    Torch/transformers are imported HERE, not at module top, so the data gate
    can still run (and skip loudly) on a laptop without those packages.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from src.data_torch import (
        VerdictTagDataset,
        compute_class_weights,
        make_collate_fn,
    )
    from src.model import MultiTaskClassifier
    from src.train import evaluate_split, train_one_epoch

    collate = make_collate_fn(tokenizer.pad_token_id)
    train_ds = VerdictTagDataset(train_df, tokenizer, maps, int(cfg["max_length"]))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]),
                              shuffle=True, collate_fn=collate)
    model = MultiTaskClassifier(cfg["model_name"], maps["num_verdicts"],
                                maps["num_tags"]).to(device)
    verdict_w, tag_pos_w = compute_class_weights(
        train_df, maps, float(cfg["max_class_weight"]))
    verdict_w, tag_pos_w = verdict_w.to(device), tag_pos_w.to(device)
    head_params = list(model.verdict_head.parameters()) + list(model.tag_head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": list(model.encoder.parameters()),
          "lr": float(cfg.get("lr_encoder", cfg.get("lr", 3e-5))),
          "weight_decay": float(cfg["weight_decay"])},
         {"params": head_params,
          "lr": float(cfg.get("lr_heads", cfg.get("lr", 3e-5))),
          "weight_decay": 0.0}])
    total_steps = max(len(train_loader) * int(cfg["epochs"]), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(float(cfg["warmup_ratio"]) * total_steps),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(int(cfg["epochs"])):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               scaler, device, verdict_w, tag_pos_w,
                               float(cfg["lambda_tags"]))
        print(f"  fold {fold} epoch {epoch + 1}: train loss {loss:.4f}")

    tags_trained = float(cfg.get("lambda_tags", 0) or 0) > 0
    m = evaluate_split(model, test_df, tokenizer, maps, cfg, device,
                       with_tags=tags_trained)
    print(f"  fold {fold}: n={len(test_df)} macro-F1={m['macro_f1']:.4f}")
    return {"fold": fold, "n_rows": len(test_df),
            "macro_f1": float(m["macro_f1"]),
            "scam_recall": float(m["scam_recall"]),
            "tag_micro_f1": m["tactics"]["micro_f1"] if tags_trained else None}


def _write_gate_report(ok: bool, n_in: int, n_safe: int, n_scam: int,
                       summary: dict | None) -> None:
    """Always leave a written record, including the SKIPPED case.

    A skipped experiment must be visible in results/, not just in a console
    message that disappears when the terminal closes.
    """
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# E4 - cluster-grouped CV on real messages", "",
             f"- in-domain real (`user_real_scam` + `user_real_safe`): **{n_in}**",
             f"- `user_real_scam`: **{n_scam}**",
             f"- `user_real_safe` hard negatives: **{n_safe}**",
             "- gate: >=50 in-domain AND >=20 user_real_safe", ""]
    payload = {"gate_passed": bool(ok), "n_in_domain": n_in, "n_safe": n_safe,
               "n_scam": n_scam, "folds": [], "note": ""}
    if not ok:
        lines += ["## E4 SKIPPED", "",
                  f"**NO VALID REAL CV:** {n_in} in-domain rows cannot support a "
                  "5-fold claim. No real-message CV was run, and no mean/std is "
                  "reported.",
                  "Collect real Hinglish messages (the labeling CLI supports it), "
                  "then rerun this gate."]
        payload["note"] = "skipped: real-data gate failed"
    else:
        assert summary is not None
        payload = summary
        lines += ["| fold | rows | macro-F1 | scam recall | tag micro-F1 |",
                  "|---|---|---|---|---|"]
        for r in summary["folds"]:
            tag = "n/a" if r["tag_micro_f1"] is None else f"{r['tag_micro_f1']:.3f}"
            lines.append(f"| {r['fold']} | {r['n_rows']} | {r['macro_f1']:.3f} "
                         f"| {r['scam_recall']:.3f} | {tag} |")
        lines += ["", f"**Mean macro-F1 = {summary['mean_macro_f1']:.3f} "
                      f"+/- {summary['std_macro_f1']:.3f} (std over folds).**"]
    (out_dir / "e4_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "E4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {out_dir / 'E4.md'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="E4 cluster-grouped CV on real data")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    if args.model:
        cfg["model_name"] = args.model

    frames = load_frames()
    real = frames["test_real"]
    in_domain = real[real["real_stratum"].isin(["user_real_scam", "user_real_safe"])]
    ok, n_in, n_safe = gate_ok(real)
    n_scam = int((real["real_stratum"] == "user_real_scam").sum())
    print(f"E4 GATE: in-domain={n_in}, user_real_safe={n_safe}")
    if not ok:
        print("!" * 70)
        print("E4 GATE FAILED: need >=50 in-domain real messages and >=20 "
              "user_real_safe hard negatives. SKIPPING E4.")
        print("!" * 70)
        _write_gate_report(ok=False, n_in=n_in, n_safe=n_safe, n_scam=n_scam,
                           summary=None)
        return
    if n_scam < 20:
        print("WARNING: fewer than 20 user_real_scam rows; recall on the held-out "
              "folds will be extremely noisy.")

    import torch
    from sklearn.model_selection import GroupKFold
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    from src.data_torch import build_label_maps
    from src.train import set_seed

    hf_logging.set_verbosity_error()
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps = build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    groups = in_domain["template_cluster"].to_numpy()
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for fold, (tr_idx, te_idx) in enumerate(folds.split(in_domain, groups=groups), 1):
        # Reseed each fold so a rerun is reproducible fold by fold.
        set_seed(int(cfg["seed"]) + fold)
        # train split + the other 4 folds; test = this held-out fold
        train_df = pd.concat([frames["train"], in_domain.iloc[tr_idx]],
                             ignore_index=True)
        test_df = in_domain.iloc[te_idx].reset_index(drop=True)
        results.append(run_fold(train_df, test_df, cfg, maps, tokenizer,
                                device, fold))

    f1s = np.array([r["macro_f1"] for r in results])
    summary = {"gate_passed": True, "n_in_domain": n_in, "n_safe": n_safe,
               "n_scam": n_scam, "folds": results,
               "mean_macro_f1": float(f1s.mean()),
               "std_macro_f1": float(f1s.std())}
    out = PROJECT_ROOT / "results" / "e4_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_gate_report(ok=True, n_in=n_in, n_safe=n_safe, n_scam=n_scam,
                       summary=summary)
    print(f"\nE4 mean macro-F1 = {f1s.mean():.4f} +/- {f1s.std():.4f}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
