"""
src/train.py - fine-tune a multilingual transformer (Phase 3).

THE RULES THIS FILE FOLLOWS:
  * Train on the TRAIN split only (public_weak excluded, same as baselines).
  * Choose the best checkpoint using VALIDATION ONLY, and specifically the
    mean of val_synthetic macro-F1 and val_real balanced accuracy (Phase 3.6).
    Combined-val macro-F1 is still logged, but only for observation: real val
    is 98.3% scam, so it rewards a model that flags everything.
  * Log test_a / test_b / test_real every epoch for OBSERVATION, but never use
    them for checkpoint selection or hyperparameter decisions.
  * test_real is logged PER STRATUM, never as one blended number.
  * "suspicious" is not a class (Phase 2.5). This is binary scam/safe.

USAGE
  python src/train.py --config configs/train.yaml \
      --model google/muril-base-cased --run_name e1_muril

Everything a run produced lands in outputs/<run_name>/:
  best_model.pt, tokenizer/, label_maps.json, config_used.yaml,
  training_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from src.config import PROJECT_ROOT, REAL_STRATA, SEED
from src.data_torch import (
    VerdictTagDataset,
    build_label_maps,
    class_weight_summary,
    compute_class_weights,
    make_collate_fn,
    save_label_maps,
)
from src.evaluate import tactic_metrics, tag_support, verdict_metrics
from src.io_utils import load_messages
from src.model import MultiTaskClassifier, multi_task_loss

hf_logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# reproducibility / config
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (including CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_train_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def selection_score(val_synth_macro_f1: float | None,
                    val_real_balanced_accuracy: float | None) -> float:
    """The VALIDATION-ONLY number Phase 3.6 picks the best checkpoint with.

    It is the MEAN of two balanced validation numbers:
      * val_synthetic macro-F1 - synthetic val is ~47% scam, so macro-F1 does
        not reward flagging everything;
      * val_real balanced accuracy - (scam recall + safe recall) / 2 on the
        real validation rows. Real val is 98.3% scam, so combined macro-F1
        there is almost all scam recall; balanced accuracy makes the ~60 real
        ham rows count as much as the 3,511 real scams.

    Why the change: the old combined-val macro-F1 rewarded over-flagging, which
    is exactly the failure mode the real ham strata keep showing. A model that
    says "scam" to everything scored ~0.944 on combined val.

    Only validation values are accepted - the function has no test parameter,
    so it is structurally impossible to select on test data. If one part is
    missing (no real val rows), the other is used alone and that is logged.
    """
    parts = []
    for value in (val_synth_macro_f1, val_real_balanced_accuracy):
        if value is None:
            continue
        number = float(value)
        if math.isnan(number):
            continue
        parts.append(number)
    return sum(parts) / len(parts) if parts else 0.0


def select_best_checkpoint(val_metric: float, best_metric: float) -> bool:
    """Return True if the current VALIDATION score is the new best.

    `val_metric` is the output of selection_score() - validation only. Test
    metrics are not passed in anywhere, which makes it impossible to select on
    test by accident.
    """
    return val_metric > best_metric


def _accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_frames(mix: str = "mix_synth_plus_real") -> dict[str, pd.DataFrame]:
    """Load every split once for one training mix.

    mix_synth_only      : train/val use synthetic rows only
    mix_synth_plus_real : train/val also include real external + real ham
    """
    df = load_messages()
    if "suspicious" in set(df["verdict"]):
        raise SystemExit("suspicious rows in the main table - run src.build_dataset")
    synthetic = df["source"].isin(["synthetic", "synthetic_llm"])
    # Phase 3.5: UCI public spam (public_weak) is NOT used for training. Its
    # legitimate ham is loaded separately as public_ham_eval, and its spam is
    # easy English spam that would flatter the model. Guard here too, in case
    # an older messages.parquet still contains those rows.
    not_weak = df["source"] != "public_weak"
    train = df[(df["split"] == "train") & not_weak]
    val = df[(df["split"] == "val") & not_weak]
    if mix == "mix_synth_only":
        train = train[synthetic.loc[train.index]]
        val = val[synthetic.loc[val.index]]
    elif mix != "mix_synth_plus_real":
        raise ValueError(f"unknown train mix {mix!r}")
    frames = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "val_synthetic": val[val["source"].isin(
            ["synthetic", "synthetic_llm"])].reset_index(drop=True),
        "val_real": val[val["source"].isin(
            ["external_reported", "public_ham_eval"])].reset_index(drop=True),
        "test_a": df[df["split"] == "test_a"].reset_index(drop=True),
        "test_b": df[df["split"] == "test_b"].reset_index(drop=True),
        "test_real": df[df["split"] == "test_real"].reset_index(drop=True),
    }
    print(f"  rows: { {k: len(v) for k, v in frames.items()} } (mix={mix})")
    return frames


def make_loader(df, tokenizer, maps, cfg, shuffle: bool) -> DataLoader:
    ds = VerdictTagDataset(df, tokenizer, maps, max_length=int(cfg["max_length"]))
    collate = make_collate_fn(tokenizer.pad_token_id)
    return DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]) if shuffle else int(cfg["eval_batch_size"]),
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=int(cfg.get("num_workers", 0)),
    )


# ---------------------------------------------------------------------------
# train / evaluate one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    verdict_w, tag_pos_w, lambda_tags: float,
                    use_amp: bool | None = None) -> float:
    """One pass over the training data. Returns mean total loss.

    use_amp=None means "use mixed precision automatically on CUDA". The caller
    passes an explicit value so configs/train.yaml fp16:false is respected.
    """
    model.train()
    total, n = 0.0, 0
    if use_amp is None:
        use_amp = device.type == "cuda"
    use_amp = bool(use_amp and device.type == "cuda")
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            verdict_logits, tag_logits = model(batch["input_ids"],
                                               batch["attention_mask"])
            loss, _ = multi_task_loss(
                verdict_logits, tag_logits, batch["verdict"], batch["tags"],
                verdict_weights=verdict_w, tag_pos_weight=tag_pos_w,
                lambda_tags=lambda_tags)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        total += float(loss.detach()) * len(batch["verdict"])
        n += len(batch["verdict"])
    return total / max(n, 1)


@torch.no_grad()
def predict_arrays(model, loader, device, maps) -> dict:
    """Run inference and return numpy arrays: labels, probs, tag probs."""
    model.eval()
    verdict_ids, probs, tag_probs = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        verdict_logits, tag_logits = model(batch["input_ids"], batch["attention_mask"])
        probs.append(torch.softmax(verdict_logits, dim=-1).cpu().numpy())
        tag_probs.append(torch.sigmoid(tag_logits).cpu().numpy())
        verdict_ids.extend(batch["verdict"].cpu().tolist())
    p = np.concatenate(probs, axis=0) if probs else np.zeros((0, maps["num_verdicts"]))
    t = np.concatenate(tag_probs, axis=0) if tag_probs else np.zeros((0, maps["num_tags"]))
    return {
        "y_true": [maps["id_to_verdict"][str(i)] for i in verdict_ids],
        "verdict_prob": p,
        "tag_prob": t,
        "y_pred": [maps["id_to_verdict"][str(i)] for i in p.argmax(axis=1)],
    }


def evaluate_split(model, df, tokenizer, maps, cfg, device,
                   with_tags: bool = True) -> dict:
    """Metrics for one split (used for val + observation-only test logging).

    `with_tags=False` for the E2 ablation (lambda_tags=0): the tag head exists
    but never received a gradient, so scoring it would invent a fake number.
    """
    if df.empty:
        return {}
    loader = make_loader(df, tokenizer, maps, cfg, shuffle=False)
    out = predict_arrays(model, loader, device, maps)
    m = verdict_metrics(df, out["y_pred"])
    if with_tags:
        # Predicted tactic tags: probability >= 0.5.
        pred_tags = []
        for row in out["tag_prob"]:
            pred_tags.append([maps["id_to_tactic"][str(i)] for i, v in enumerate(row)
                              if v >= 0.5])
        m["tactics"] = tactic_metrics(df, pred_tags)
    else:
        m["tactics"] = {"micro_f1": None, "macro_f1": None, "per_tag_f1": {},
                        "headline_tags": [], "rows_with_gold_tags": 0,
                        "available": False}
    m["accuracy"] = _accuracy(out["y_true"], out["y_pred"])
    return m


def evaluate_val_loss(model, loader, device, verdict_w, tag_pos_w,
                      lambda_tags: float) -> float:
    """Validation loss only (no test data involved)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            verdict_logits, tag_logits = model(batch["input_ids"],
                                               batch["attention_mask"])
            loss, _ = multi_task_loss(
                verdict_logits, tag_logits, batch["verdict"], batch["tags"],
                verdict_weights=verdict_w, tag_pos_weight=tag_pos_w,
                lambda_tags=lambda_tags)
            total += float(loss) * len(batch["verdict"])
            n += len(batch["verdict"])
    return total / max(n, 1)


def log_real_strata(model, frames, tokenizer, maps, cfg, device,
                    with_tags: bool = True) -> list[dict]:
    """Observation-only metrics for each real stratum (never blended).

    These numbers are written to the log every epoch so you can SEE them, but
    they are never passed to the checkpoint-selection function.
    """
    real = frames["test_real"]
    rows = []
    for strat in REAL_STRATA:
        sub = real[real["real_stratum"] == strat].reset_index(drop=True)
        if sub.empty:
            continue
        support = tag_support(sub)
        validated = [t for t, n in support.items() if n >= 20]
        loader = make_loader(sub, tokenizer, maps, cfg, shuffle=False)
        out = predict_arrays(model, loader, device, maps)
        m = verdict_metrics(sub, out["y_pred"])
        if with_tags:
            pred_tags = [[maps["id_to_tactic"][str(i)] for i, v in enumerate(row)
                          if v >= 0.5] for row in out["tag_prob"]]
            m["tactics"] = tactic_metrics(sub, pred_tags, validated_tags=validated)
            tag_micro = m["tactics"]["micro_f1"]
        else:
            tag_micro = None
        rows.append({"stratum": strat, "n_rows": len(sub),
                     "macro_f1": m["macro_f1"], "scam_recall": m["scam_recall"],
                     "fpr_on_safe": m["fpr_on_safe"],
                     "too_small_for_claims": len(sub) < 30,
                     "tag_micro_f1": tag_micro})
    return rows


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------
def train(config_path: str, model_name: str, run_name: str,
          overrides: dict | None = None) -> Path:
    cfg = load_train_config(config_path)
    cfg["model_name"] = model_name
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    seed = int(cfg.get("seed", SEED))
    set_seed(seed)

    # PROJECT_ROOT-relative, so the run lands in the same outputs/ folder that
    # src/evaluate.py and src/predict.py read, no matter which folder you ran
    # the command from.
    out_dir = Path(PROJECT_ROOT) / str(cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the EXACT config next to the results.
    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"(seed={seed}) src/train.py  run={run_name}")
    print("=" * 60)
    print(f"model : {model_name}")
    print(f"device: {device}"
          f"{' (fp16)' if device.type == 'cuda' and cfg.get('fp16') else ''}")
    lr_encoder = float(cfg.get("lr_encoder", cfg.get("lr", 3e-5)))
    lr_heads = float(cfg.get("lr_heads", cfg.get("lr", 3e-5)))
    mix = str(cfg.get("train_mix", "mix_synth_plus_real"))
    print(f"epochs={cfg['epochs']} batch={cfg['batch_size']} "
          f"lr_encoder={lr_encoder:g} lr_heads={lr_heads:g} "
          f"lambda_tags={cfg['lambda_tags']} mix={mix}")
    tags_trained = float(cfg.get("lambda_tags", 0) or 0) > 0
    if not tags_trained:
        print("lambda_tags=0 -> verdict-only run: tag metrics will be reported as n/a")

    frames = load_frames(mix)
    maps = build_label_maps()
    save_label_maps(out_dir / "label_maps.json", maps)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    tokenizer.save_pretrained(out_dir / "tokenizer")

    train_loader = make_loader(frames["train"], tokenizer, maps, cfg, shuffle=True)
    val_loader = make_loader(frames["val"], tokenizer, maps, cfg, shuffle=False)

    model = MultiTaskClassifier(model_name, maps["num_verdicts"],
                                maps["num_tags"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")
    verdict_w, tag_pos_w = compute_class_weights(
        frames["train"], maps, float(cfg["max_class_weight"]))
    verdict_w, tag_pos_w = verdict_w.to(device), tag_pos_w.to(device)
    print("class balance:", class_weight_summary(frames["train"], maps))

    # Separate parameter groups: slow encoder, fast new heads.
    head_params = list(model.verdict_head.parameters()) + list(model.tag_head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": list(model.encoder.parameters()), "lr": lr_encoder,
          "weight_decay": float(cfg["weight_decay"])},
         {"params": head_params, "lr": lr_heads, "weight_decay": 0.0}])
    total_steps = max(len(train_loader) * int(cfg["epochs"]), 1)
    warmup_steps = int(float(cfg["warmup_ratio"]) * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    amp_enabled = device.type == "cuda" and bool(cfg.get("fp16"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = -1.0
    best_epoch = -1
    bad_epochs = 0
    log_path = out_dir / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()          # fresh log for this run

    print("\nepoch | train_loss | val_loss | select_score | val_tagF1 | test_a | test_b | time")
    print("-" * 84)
    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        try:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device,
                verdict_w, tag_pos_w, float(cfg["lambda_tags"]),
                use_amp=amp_enabled)
        except torch.cuda.OutOfMemoryError:
            print("\nCUDA OUT OF MEMORY. Lower batch_size (try 8 or 4) and rerun.")
            print("Nothing was written as a 'best' checkpoint.")
            raise

        val_loss = evaluate_val_loss(model, val_loader, device, verdict_w,
                                     tag_pos_w, float(cfg["lambda_tags"]))
        val_metrics = evaluate_split(model, frames["val"], tokenizer, maps, cfg,
                                    device, with_tags=tags_trained)
        val_synth = (evaluate_split(model, frames["val_synthetic"], tokenizer,
                                    maps, cfg, device, with_tags=tags_trained)
                     if not frames["val_synthetic"].empty else {})
        val_real = (evaluate_split(model, frames["val_real"], tokenizer,
                                   maps, cfg, device, with_tags=tags_trained)
                    if not frames["val_real"].empty else {})

        # Observation-only metrics. NEVER used for checkpoint selection.
        obs = {}
        for split in ("test_a", "test_b"):
            obs[split] = evaluate_split(model, frames[split], tokenizer, maps, cfg,
                                        device, with_tags=tags_trained)
        obs["test_real_strata"] = log_real_strata(
            model, frames, tokenizer, maps, cfg, device, with_tags=tags_trained)

        # Phase 3.6 selection: mean(synthetic-val macro-F1, real-val balanced
        # accuracy). Both numbers are logged next to the score, and neither
        # comes from a test split.
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            # Observation-only combined-val macro-F1 - NOT the selector.
            "val_macro_f1": float(val_metrics.get("macro_f1") or 0.0),
            "val_synth_macro_f1": float(val_synth.get("macro_f1") or 0.0),
            "val_real_macro_f1": float(val_real.get("macro_f1") or 0.0),
            "val_real_balanced_accuracy": val_real.get("balanced_accuracy"),
            "selection_score": 0.0,
            "val_tag_micro_f1": (float(val_metrics["tactics"]["micro_f1"])
                                 if tags_trained else None),
            "test_a_macro_f1": obs["test_a"].get("macro_f1"),
            "test_b_macro_f1": obs["test_b"].get("macro_f1"),
            "test_real_strata": obs["test_real_strata"],
            "train_mix": mix,
            "tags_trained": tags_trained,
            "seconds": round(time.time() - t0, 1),
        }
        row["selection_score"] = selection_score(
            row["val_synth_macro_f1"],
            None if frames["val_real"].empty else row["val_real_balanced_accuracy"])
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        tag_cell = "n/a" if not tags_trained else f"{row['val_tag_micro_f1']:.4f}"
        def _fmt(v):
            return "n/a" if v is None else f"{v:.4f}"
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
              f"{row['selection_score']:12.4f} | {tag_cell:>9} | "
              f"{_fmt(row['test_a_macro_f1']):>7} | "
              f"{_fmt(row['test_b_macro_f1']):>7} | {row['seconds']}s")
        print(f"      selection = mean(val_synthetic macro-F1="
              f"{_fmt(row['val_synth_macro_f1'])}, val_real balanced-acc="
              f"{_fmt(row['val_real_balanced_accuracy'])}); "
              f"combined val macro-F1={_fmt(row['val_macro_f1'])} (observation)")
        for s in obs["test_real_strata"]:
            flag = "  <-- TOO SMALL" if s.get("too_small_for_claims") else ""
            print(f"    OBSERVE test_real/{s['stratum']}: n={s['n_rows']} "
                  f"recall={s['scam_recall']:.4f} FPR={s['fpr_on_safe']:.4f} "
                  f"macro-F1={_fmt(s['macro_f1'])} (selection: val only){flag}")

        # ---- checkpoint selection: VALIDATION ONLY ---------------------
        if select_best_checkpoint(row["selection_score"], best_val):
            best_val = row["selection_score"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / cfg["checkpoint_name"])
            (out_dir / "best_epoch.json").write_text(
                json.dumps({
                    "epoch": epoch,
                    "selection_score": best_val,
                    "selection_metric": "mean(val_synthetic_macro_f1, "
                                        "val_real_balanced_accuracy)",
                    "val_synthetic_macro_f1": row["val_synth_macro_f1"],
                    "val_real_balanced_accuracy": row["val_real_balanced_accuracy"],
                }, indent=2))
            print(f"    NEW BEST (validation selection score={best_val:.4f}) "
                  "-> saved checkpoint")
        else:
            bad_epochs += 1
            print(f"    no improvement ({bad_epochs}/{cfg['patience']})")
            if bad_epochs >= int(cfg["patience"]):
                print(f"    early stopping at epoch {epoch} (patience reached)")
                break

    # Machine-readable run summary for the notebooks/benchmark. It records
    # that checkpoint selection used VALIDATION ONLY.
    run_meta = {
        "run_name": run_name,
        "model_name": model_name,
        "tags_trained": tags_trained,
        "lambda_tags": float(cfg["lambda_tags"]),
        "train_mix": mix,
        "lr_encoder": lr_encoder,
        "lr_heads": lr_heads,
        "n_train_rows": int(len(frames["train"])),
        "n_val_rows": int(len(frames["val"])),
        "n_val_synthetic": int(len(frames["val_synthetic"])),
        "n_val_real": int(len(frames["val_real"])),
        "trainable_parameters": int(n_params),
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_selection_score": best_val,
        "selection_metric": ("mean(val_synthetic macro-F1, "
                             "val_real balanced accuracy)"),
        "selection_uses_test_data": False,
        "early_stopped": bad_epochs >= int(cfg["patience"]),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"\nDONE. best epoch={best_epoch} validation selection score={best_val:.4f}")
    print("Checkpoint selection used VALIDATION ONLY "
          "(mean of val_synthetic macro-F1 and val_real balanced accuracy); "
          "test sets were observation only.")
    print(f"outputs: {out_dir}")
    return out_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer (Phase 3)")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--run_name", default="e1_muril")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_tags", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_mix", default=None,
                        choices=["mix_synth_only", "mix_synth_plus_real"],
                        help="which training mix to use")
    args = parser.parse_args(argv)
    base = load_train_config(args.config)
    overrides = {"epochs": args.epochs, "batch_size": args.batch_size,
                 "lr": args.lr, "lambda_tags": args.lambda_tags,
                 "seed": args.seed, "train_mix": args.train_mix}
    # Legacy --lr still works, but it would overwrite BOTH groups. Only apply
    # it when the caller explicitly passes it.
    if args.lr is not None:
        overrides["lr_encoder"] = args.lr
        overrides["lr_heads"] = args.lr
    train(args.config, args.model or base["model_name"], args.run_name,
          overrides=overrides)


if __name__ == "__main__":
    main()
