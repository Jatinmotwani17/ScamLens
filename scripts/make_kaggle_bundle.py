"""
scripts/make_kaggle_bundle.py - build scamlens_bundle.zip for Kaggle / Colab.

WHY A BUNDLE
  Kaggle notebooks are offline-ish and slow at downloading 9 MB of public
  corpora. The bundle therefore carries the ONE artefact training cannot
  rebuild cheaply: data/processed/messages.parquet (already deduplicated,
  clustered and split, with `split` + `real_stratum` inside).

WHAT GOES IN
  src/ configs/ tests/ notebooks/ requirements.txt README.md
  data/processed/messages.parquet
  plus, when they exist, outputs/<run>/label_maps.json and config_used.yaml
  copied to run_metadata/<run>/ (the label map is derived from configs, but
  shipping it lets a notebook verify the vocabulary without torch).

WHAT NEVER GOES IN
  data/private, data/interim, data/external, venv, outputs/, models/,
  checkpoints/, .git, __pycache__, and every weight file
  (*.pt *.bin *.safetensors *.ckpt).

VERIFY
  `--verify` unzips the bundle into a temp folder and (a) loads the parquet,
  (b) runs `src.train.load_frames` for both mixes, (c) runs
  `src.run_experiments --dry_run`. `--train_smoke` additionally trains one
  epoch of sshleifer/tiny-distilbert-base-cased on a SUBSAMPLED COPY of the
  extracted data, so the training loop itself is exercised on CPU in a couple
  of minutes. The bundle on disk is never modified.

Run:
  python scripts/make_kaggle_bundle.py
  python scripts/make_kaggle_bundle.py --verify
  python scripts/make_kaggle_bundle.py --verify --train_smoke
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE_DIRS = ("src", "configs", "tests", "notebooks")
INCLUDE_FILES = ("requirements.txt", "README.md")
# Added explicitly, not by walking data/, because data/ also holds private rows.
DATA_FILES = ("data/processed/messages.parquet",)
# Per-run label map + config: small, text only, no weights.
RUN_META_FILES = ("label_maps.json", "config_used.yaml")

EXCLUDE_DIR_PARTS = {
    ".git", "venv", ".venv", "env", "outputs", "models", "checkpoints",
    "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "htmlcov",
    "data",     # every data/ subfolder is private, raw or regenerable
}
EXCLUDE_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt",
                    ".joblib", ".pyc", ".pyo"}

# Files that must never be inside the zip, used by --verify.
FORBIDDEN_PREFIXES = ("data/private/", "data/interim/", "data/external/",
                      "venv/", "outputs/", "models/", "checkpoints/", ".git/")
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".joblib")


def collect_entries() -> list[tuple[Path, str]]:
    """Return (source file, path inside the zip) for everything to ship."""
    entries: list[tuple[Path, str]] = []

    for folder in INCLUDE_DIRS:
        base = ROOT / folder
        if not base.exists():
            raise SystemExit(f"missing folder: {base}")
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDE_DIR_PARTS for part in rel.parts):
                continue
            if path.suffix in EXCLUDE_SUFFIXES:
                continue
            entries.append((path, rel.as_posix()))

    for name in INCLUDE_FILES:
        path = ROOT / name
        if path.exists():
            entries.append((path, name))

    for name in DATA_FILES:
        path = ROOT / name
        if not path.exists():
            raise SystemExit(
                f"{name} is missing - run `python -m src.build_dataset` first.")
        entries.append((path, name))

    # Optional run metadata (label maps / configs). Text only, still no weights.
    outputs = ROOT / "outputs"
    if outputs.exists():
        for run in sorted(p for p in outputs.iterdir() if p.is_dir()):
            for name in RUN_META_FILES:
                path = run / name
                if path.exists():
                    entries.append((path, f"run_metadata/{run.name}/{name}"))

    return entries


def build(out: Path) -> Path:
    if out.exists():
        out.unlink()
    entries = collect_entries()
    if out.is_absolute():
        out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in entries:
            zf.write(src, arcname)

    size_mb = out.stat().st_size / 1e6
    print(f"WROTE {out} ({len(entries)} files, {size_mb:.2f} MB)")
    groups: dict[str, int] = {}
    for _, arcname in entries:
        key = arcname.split("/")[0]
        groups[key] = groups.get(key, 0) + 1
    print("  contents:", ", ".join(f"{k} ({v})" for k, v in sorted(groups.items())))
    print("  excluded : data/private, data/interim, data/external, venv, "
          "outputs, models, checkpoints, .git, all weights")
    return out


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
CHECK_CODE = """
import pandas as pd
from src.config import TACTIC_TAGS
from src.data_torch import build_label_maps
from src.train import load_frames

df = pd.read_parquet("data/processed/messages.parquet")
print("  parquet rows:", len(df))
print("  splits:", df["split"].value_counts().to_dict())
maps = build_label_maps()
assert maps["num_tags"] == len(TACTIC_TAGS), maps.keys()
assert maps["verdict_classes"] == ["safe", "scam"], maps["verdict_classes"]
for mix in ("mix_synth_only", "mix_synth_plus_real"):
    frames = load_frames(mix)
    assert len(frames["train"]) and len(frames["val"]), mix
    print(f"  {mix}: train={len(frames['train'])}, val={len(frames['val'])}, "
          f"val_real={len(frames['val_real'])}, test_real={len(frames['test_real'])}")
print("  label maps + training frames OK")
"""

TRIM_CODE = """
import pandas as pd

path = "data/processed/messages.parquet"
df = pd.read_parquet(path)
targets = {"train": 500, "val": 400, "test_a": 120, "test_b": 120,
           "test_real": 200}
parts = []
for split, n in targets.items():
    sub = df[df["split"] == split]
    if len(sub) <= n:
        parts.append(sub)
        continue
    taken = []
    for _, group in sub.groupby("verdict"):
        k = max(1, int(round(n * len(group) / len(sub))))
        taken.append(group.sample(n=min(k, len(group)), random_state=42))
    parts.append(pd.concat(taken))
pd.concat(parts).to_parquet(path, index=False)
print("  trimmed a COPY for the smoke run:",
      pd.read_parquet(path)["split"].value_counts().to_dict())
"""


def _run(cmd: list[str], cwd: Path) -> None:
    print("  $ " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)          # module imports work in the copy
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def verify(zip_path: Path, train_smoke: bool = False) -> None:
    """Unzip into a temp dir and prove the code runs WITHOUT data/external."""
    tmp = Path(tempfile.mkdtemp(prefix="scamlens_bundle_"))
    print(f"\nVERIFY: extracting to {tmp}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
        names = zf.namelist()

    bad = [n for n in names
           if n.startswith(FORBIDDEN_PREFIXES) or n.endswith(FORBIDDEN_SUFFIXES)]
    if bad:
        raise SystemExit(f"FORBIDDEN FILES IN BUNDLE: {bad[:10]}")
    if (tmp / "data" / "external").exists():
        raise SystemExit("data/external leaked into the bundle")
    print(f"  {len(names)} files; no private/raw/weights; data/external absent")

    print("\nVERIFY: data + label maps + train frames")
    _run([sys.executable, "-c", CHECK_CODE], cwd=tmp)

    print("\nVERIFY: src.run_experiments --dry_run (model ids, gate, commands)")
    _run([sys.executable, "-m", "src.run_experiments", "--dry_run"], cwd=tmp)

    if train_smoke:
        print("\nVERIFY: one real epoch of src.train on a subsampled COPY")
        _run([sys.executable, "-c", TRIM_CODE], cwd=tmp)
        # A tiny DISTILBERT, not prajjwal1/bert-tiny: that repo ships no fast
        # tokenizer file, and the slow->fast conversion fails on current
        # transformers even with sentencepiece installed. The point of the
        # smoke run is the training loop, not the model.
        _run([sys.executable, "-m", "src.train",
              "--config", "configs/train.yaml",
              "--model", "sshleifer/tiny-distilbert-base-cased",
              "--run_name", "bundle_smoke",
              "--epochs", "1",
              "--batch_size", "8",
              "--lr", "0.0002"], cwd=tmp)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nVERIFY PASSED: the bundle runs from an unzipped copy with "
          "data/external absent.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build scamlens_bundle.zip")
    parser.add_argument("--out", default=str(ROOT / "scamlens_bundle.zip"),
                        help="output zip path (default: project root)")
    parser.add_argument("--verify", action="store_true",
                        help="unzip to a temp dir and run the checks")
    parser.add_argument("--train_smoke", action="store_true",
                        help="with --verify, also train 1 epoch of bert-tiny "
                             "on a subsampled copy of the extracted data")
    args = parser.parse_args(argv)

    zip_path = Path(args.out)
    build(zip_path)
    if args.verify:
        verify(zip_path, train_smoke=args.train_smoke)


if __name__ == "__main__":
    main()
