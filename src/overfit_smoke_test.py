"""
src/overfit_smoke_test.py - the "can it learn at all?" test.

WHY: before spending an hour on real training, take just 16 training examples
and train for 30 small steps. A working model should MEMORISE them, so the
verdict loss must drop by at least 70%. If it does not, something is broken
(wrong labels, frozen encoder, bad loss, tokenizer mismatch) and real training
would waste your time.

ONE HONEST DETAIL ABOUT THE LEARNING RATE
-----------------------------------------
The requested production LR is 5e-5. Thirty steps at 5e-5 physically cannot
move a freshly initialised 768->2 head far enough to memorise anything: the
total weight movement is about 30 * 5e-5 = 0.0015. So the smoke test uses the
production LR for the pretrained ENCODER (5e-5) and a larger LR for the two
NEW HEADS (5e-5 * head_lr_mult, default 200 = 0.01), which is standard practice for
randomly initialised heads. This is a WIRING test, not a tuning experiment.
Run `--strict_lr` to use 5e-5 everywhere and see the difference.

The PASS criterion is measured in eval() mode on the same 16 examples (dropout
off), so it compares like with like instead of chasing noisy training batches.

Run on a laptop (CPU is fine, about 5 minutes):
    python -m src.overfit_smoke_test
    python -m src.overfit_smoke_test --model google/muril-base-cased --steps 30
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.config import PROJECT_ROOT, SEED
from src.data_torch import (
    VerdictTagDataset,
    build_label_maps,
    compute_class_weights,
    make_collate_fn,
)
from src.io_utils import load_messages
from src.model import MultiTaskClassifier, multi_task_loss

hf_logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so the smoke test is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_sixteen(df, seed: int):
    """8 scam + 8 safe training rows, so both classes are represented."""
    rng = np.random.default_rng(seed)
    scam = df[df["verdict"] == "scam"]
    safe = df[df["verdict"] == "safe"]
    idx = list(rng.choice(scam.index, 8, replace=False)) + \
          list(rng.choice(safe.index, 8, replace=False))
    return df.loc[idx].reset_index(drop=True)


@torch.no_grad()
def evaluate_on_tiny(model, batch, verdict_w, tag_pos_w, lambda_tags: float) -> dict:
    """Loss + accuracy on the SAME 16 rows, with dropout switched OFF."""
    model.eval()
    verdict_logits, tag_logits = model(batch["input_ids"], batch["attention_mask"])
    loss, parts = multi_task_loss(
        verdict_logits, tag_logits, batch["verdict"], batch["tags"],
        verdict_weights=verdict_w, tag_pos_weight=tag_pos_w,
        lambda_tags=lambda_tags)
    pred = verdict_logits.argmax(dim=-1)
    acc = float((pred == batch["verdict"]).float().mean())
    model.train()
    return {"loss": float(loss), "verdict_loss": float(parts["verdict_loss"]),
            "tag_loss": float(parts["tag_loss"]), "accuracy": acc}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="16-example overfit smoke test")
    parser.add_argument("--model", default=None)
    parser.add_argument("--steps", type=int, default=30,
                        help="30 by default (the Phase 3 spec)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="LR for the pretrained encoder")
    parser.add_argument("--head_lr_mult", type=float, default=200.0,
                        help="new heads get lr * this (they start random)")
    parser.add_argument("--strict_lr", action="store_true",
                        help="use --lr for the heads too (usually FAILS; kept for honesty)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = 8 on CPU, 16 on GPU")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "train.yaml").read_text())
    model_name = args.model or cfg["model_name"]
    max_length = int(args.max_length or cfg["max_length"])
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or (16 if device.type == "cuda" else 8)
    head_mult = 1.0 if args.strict_lr else args.head_lr_mult
    head_lr = args.lr * head_mult

    print(f"(seed={args.seed}) src/overfit_smoke_test.py")
    print("=" * 60)
    print(f"model     : {model_name}")
    print(f"device    : {device}")
    print(f"steps     : {args.steps}  batch: {batch_size}")
    print(f"encoder LR: {args.lr:g}")
    print(f"head LR   : {head_lr:g}  (encoder LR x {head_mult:g})")
    if head_mult == 1.0:
        print("NOTE: --strict_lr uses 5e-5 on random heads; 30 steps may be too few.")
    print("Loading 16 training messages and the tokenizer ...")

    df = load_messages()
    train = df[(df["split"] == "train") & (df["source"] != "public_weak")]
    tiny = pick_sixteen(train, args.seed)
    maps = build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    ds = VerdictTagDataset(tiny, tokenizer, maps, max_length=max_length)
    collate = make_collate_fn(tokenizer.pad_token_id)
    full_batch = collate([ds[i] for i in range(len(ds))])
    full_batch = {k: v.to(device) for k, v in full_batch.items()}
    # Mini-batches for the optimizer; the PASS check always uses all 16.
    mini_batches = []
    for start in range(0, len(ds), batch_size):
        chunk = collate([ds[i] for i in range(start, min(start + batch_size, len(ds)))])
        mini_batches.append({k: v.to(device) for k, v in chunk.items()})

    model = MultiTaskClassifier(model_name, maps["num_verdicts"],
                                maps["num_tags"]).to(device)
    verdict_w, tag_pos_w = compute_class_weights(train, maps,
                                                 float(cfg["max_class_weight"]))
    verdict_w, tag_pos_w = verdict_w.to(device), tag_pos_w.to(device)
    lambda_tags = float(cfg["lambda_tags"])

    # Separate parameter groups: pretrained encoder vs. new heads.
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.verdict_head.parameters()) + list(model.tag_head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.lr, "weight_decay": 0.01},
         {"params": head_params, "lr": head_lr, "weight_decay": 0.0}])

    start = evaluate_on_tiny(model, full_batch, verdict_w, tag_pos_w, lambda_tags)
    print(f"\nstep   0/{args.steps}  eval verdict loss = {start['verdict_loss']:.4f}  "
          f"total = {start['loss']:.4f}  acc = {start['accuracy']:.0%}")

    model.train()
    history = []
    for step in range(1, args.steps + 1):
        batch = mini_batches[(step - 1) % len(mini_batches)]
        optimizer.zero_grad(set_to_none=True)
        verdict_logits, tag_logits = model(batch["input_ids"], batch["attention_mask"])
        loss, parts = multi_task_loss(
            verdict_logits, tag_logits, batch["verdict"], batch["tags"],
            verdict_weights=verdict_w, tag_pos_weight=tag_pos_w,
            lambda_tags=lambda_tags)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(parts["verdict_loss"]))

        # Print every 5 steps, exactly as requested.
        if step % 5 == 0 or step == args.steps or step == 1:
            snap = evaluate_on_tiny(model, full_batch, verdict_w, tag_pos_w, lambda_tags)
            print(f"step {step:3d}/{args.steps}  train verdict = {parts['verdict_loss']:.4f}  "
                  f"eval verdict = {snap['verdict_loss']:.4f}  "
                  f"tag = {parts['tag_loss']:.4f}  acc = {snap['accuracy']:.0%}")

    end = evaluate_on_tiny(model, full_batch, verdict_w, tag_pos_w, lambda_tags)
    drop = 1.0 - (end["verdict_loss"] / max(start["verdict_loss"], 1e-9))
    print("\n" + "-" * 62)
    print(f"eval verdict loss before : {start['verdict_loss']:.4f}")
    print(f"eval verdict loss after  : {end['verdict_loss']:.4f}")
    print(f"drop                     : {drop:.1%} (need >= 70%)")
    print(f"accuracy on the 16 rows  : {end['accuracy']:.0%}")
    if drop >= 0.70:
        print("OVERFIT SMOKE TEST: PASS - the data/label/loss path can learn.")
        return 0
    print("OVERFIT SMOKE TEST: FAIL - do not start real training yet.")
    print("  Check: labels, tokenizer, learning rate, frozen encoder, loss weights.")
    print("  If you used --strict_lr, retry without it: 30 steps at 5e-5 on")
    print("  random heads is usually too small an update.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
