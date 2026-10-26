"""
src/benchmark.py - size + CPU speed, because Phase 5 has to serve this.

For every trained transformer run and for the TF-IDF verdict baseline it
reports:
  * parameter count (trainable model weights)
  * size on disk (MB) of the artefact you would actually ship
  * CPU latency for ONE message: median and p95 over 200 messages, after
    10 warmup runs

The latency sample is drawn from `test_b` (synthetic, unseen families) so it
contains English, roman Hinglish, Devanagari and mixed script. Latency depends
on sequence length: a short SMS is faster than a 128-token message.

Run:  python -m src.benchmark
      python -m src.benchmark --n 200
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.config import PROJECT_ROOT, SEED, data_path
from src.data_torch import load_label_maps
from src.io_utils import load_messages
from src.model import MultiTaskClassifier

hf_logging.set_verbosity_error()


def disk_mb(path: Path) -> float:
    """Total size in MB of a file or a directory tree."""
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def latency_stats(fn, messages: list[str], warmup: int = 10) -> tuple[float, float]:
    """Run fn(message) once per message; return (median_ms, p95_ms)."""
    for text in messages[:warmup]:
        fn(text)                       # warmup fills caches / lazy kernels
    times = []
    for text in messages:
        t0 = time.perf_counter()
        fn(text)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def tfidf_feature_count(pipe) -> int:
    """How many TF-IDF columns the vectorizer learned (not model weights)."""
    try:
        union = pipe.named_steps["tfidf"]
        return int(sum(len(v.vocabulary_) for _, v in union.transformer_list))
    except Exception:
        return 0


def benchmark_tfidf(messages: list[str]) -> list[str]:
    path = Path(data_path("models_dir")) / "verdict_lr.joblib"
    if not path.exists():
        return ["| TF-IDF + LR | n/a | n/a | n/a | n/a | model not trained |"]
    bundle = joblib.load(path)
    pipe = bundle["pipeline"]
    clf = pipe.named_steps["clf"]
    params = int(clf.coef_.size + clf.intercept_.size)
    feats = tfidf_feature_count(pipe)
    med, p95 = latency_stats(lambda t: pipe.predict([t]), messages)
    return [f"| TF-IDF + LR | {params:,} | {disk_mb(path):.2f} | {med:.2f} | {p95:.2f} "
            f"| {feats:,} TF-IDF features |"]


def benchmark_runs(messages: list[str]) -> list[str]:
    outputs = Path(PROJECT_ROOT) / "outputs"
    if not outputs.exists():
        return []
    rows = []
    device = torch.device("cpu")       # Phase 5 CPU latency is what matters
    for run in sorted(p for p in outputs.iterdir() if p.is_dir()):
        cfg_path = run / "config_used.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        ckpt = run / cfg.get("checkpoint_name", "best_model.pt")
        if not ckpt.exists():
            continue
        maps = load_label_maps(run / "label_maps.json")
        tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
        model = MultiTaskClassifier(cfg["model_name"], maps["num_verdicts"],
                                    maps["num_tags"])
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device).eval()
        params = sum(p.numel() for p in model.parameters())
        tags = float(cfg.get("lambda_tags", 0) or 0) > 0

        @torch.no_grad()
        def predict_one(text: str, max_length=int(cfg["max_length"])):
            enc = tokenizer(text, truncation=True, max_length=max_length,
                            return_tensors="pt")
            model(enc["input_ids"], enc["attention_mask"])

        med, p95 = latency_stats(predict_one, messages)
        rows.append(f"| {run.name} ({cfg['model_name']}) | {params:,} "
                    f"| {disk_mb(ckpt):.2f} | {med:.2f} | {p95:.2f} "
                    f"| tags {'trained' if tags else 'NOT trained'} |")
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark size + CPU latency")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args(argv)

    df = load_messages()
    # Hinglish-ish, multi-style pool. test_b is synthetic and has every style.
    pool = df[df["split"] == "test_b"]
    if len(pool) < args.n:
        pool = df[df["split"].isin(["test_a", "test_b"])]
    if len(pool) < args.n:
        pool = df[df["split"] == "train"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pool), min(args.n, len(pool)), replace=False)
    messages = pool.iloc[idx]["text"].astype(str).tolist()

    print(f"(seed={SEED}) src/benchmark.py - {len(messages)} messages")
    print(f"device: CPU ({platform.processor() or platform.machine()}), "
          f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    rows = benchmark_runs(messages) + benchmark_tfidf(messages)
    lines = ["# ScamLens Phase 3 - model benchmark", "",
             f"CPU single-message latency over {len(messages)} `test_b` messages "
             "(median / p95), after 10 warmup runs.",
             f"Environment: Python {platform.python_version()}, torch {torch.__version__}, "
             f"{platform.processor() or platform.machine()}, "
             f"{torch.get_num_threads()} CPU thread(s).", "",
             "| model | parameters | disk MB | median ms | p95 ms | notes |",
             "|---|---|---|---|---|---|"] + rows
    lines += ["", "Parameters are trainable weights. Disk MB is the shipped "
                  "artefact (checkpoint for a transformer, joblib for TF-IDF). "
                  "The transformer tokenizer folder adds a few MB but is shared "
                  "across runs. Latency is single-message, no batching - Phase 5 "
                  "can batch to improve throughput."]
    out = Path(PROJECT_ROOT) / "results" / "benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
