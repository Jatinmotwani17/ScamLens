"""
scripts/scan_repo.py - fail the build if privacy or secrets leak into git.

WHAT IT SCANS
  Every file that git would consider "in the repo": tracked files, plus
  untracked files that are NOT ignored (`git ls-files -co --exclude-standard`).
  Gitignored paths (data/, models/, outputs/, .env, venv/) are skipped by git
  itself, so this cannot accidentally read your raw messages.

WHAT IT LOOKS FOR
  * Indian phone numbers (10 digits starting 6-9, with optional +91/0/spaces)
  * UPI ids / handles (name@bank)
  * email addresses
  * API keys: Google AIza..., OpenAI sk-..., AWS AKIA..., GitHub ghp_/gho_,
    Slack xox?-..., and generic "secret"/"token"/"api_key" assignments

WHAT IT PRINTS
  file:line, the pattern name, and a MASKED excerpt - never the value itself.
  Exit code 1 if any hit is found, 0 otherwise.

ALLOWLIST
  * The scanner's own file, this README-style doc, and the test fixtures that
    deliberately contain fake numbers are skipped by path.
  * A line may opt out with the comment marker `scan:allow` (use it only for
    documentation examples, never for a real number or key).

Run:
  python scripts/scan_repo.py            # scan and fail on hits
  python scripts/scan_repo.py --verbose  # also list the files scanned
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Windows/Git-Bash both work through subprocess without a shell.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Files that legitimately contain example patterns.
SKIP_PATHS = {
    "scripts/scan_repo.py",
    "DATA_LICENSES.md",
    "README.md",
    "docs/DATA_ETHICS.md",
    "docs/CLAIMS.md",
    "tests/fixtures/synthetic_seed.csv",
    "tests/test_scan_repo.py",
    # These two files DEFINE the PII patterns and test them with fake values;
    # scanning them would flag the detector itself. A real value pasted here
    # would not be caught, which is why review still matters.
    "src/scrub.py",
    "tests/test_scrub.py",
}

# Patterns are deliberately conservative: we would rather miss an exotic key
# than fail a build on a placeholder like <PHONE>.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("indian_phone", re.compile(r"(?<![\d<])(?:\+?91[\s-]?)?[6-9]\d{9}(?![\d>])")),
    ("upi_id", re.compile(r"\b[a-z][a-z0-9._-]{2,}@(?:ok(?:hdfcbank|icici|axis|sbi)|"
                          r"paytm|ybl|apl|upi|ibl|axl|barodampay)\b", re.I)),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.I)),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("slack_token", re.compile(r"\bxox[abpsr]-[A-Za-z0-9-]{10,}")),
    ("secret_assignment", re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*"
        r"['\"]?(?!\s*(?:$|none|false|true|your|xxx|placeholder|example|\.\.\.))"
        # A real secret is a literal; a function call or attribute read is not.
        r"(?!\w+\s*[(.])\S{12,}")),
]

# Obvious non-secrets that would otherwise trip `secret_assignment`.
BENIGN_SUBSTRINGS = (
    "GEMINI_API_KEY", "FREEBUFF_", "os.environ", "getenv", "secrets.token",
    "<", "${", "%s", "{", "}", "...",
)

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".json", ".toml",
                 ".cfg", ".ini", ".ipynb", ".html", ".js", ".css", ".sh",
                 ".yml", ".env", ".example", ".csv"}


def git_files() -> list[str]:
    """Tracked + untracked-but-not-ignored files, relative to the repo root."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"scan_repo: git unavailable ({exc}); scanning the working tree instead")
        return [str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                for p in REPO_ROOT.rglob("*") if p.is_file()]
    return [line.strip() for line in out.splitlines() if line.strip()]


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\x00" in chunk


def mask(value: str) -> str:
    """Show that something matched without printing it."""
    value = value.strip()
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(line_number, pattern_name, masked_excerpt)] for one file."""
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "scan:allow" in line:
            continue
        for name, pattern in PATTERNS:
            for match in pattern.finditer(line):
                value = match.group(0)
                if name == "secret_assignment" and any(
                        b in line for b in BENIGN_SUBSTRINGS):
                    continue
                hits.append((lineno, name, mask(value)))
    return hits


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scan committable files for PII/secrets")
    parser.add_argument("--verbose", action="store_true", help="list every file scanned")
    args = parser.parse_args(argv)

    files = git_files()
    scanned = 0
    findings: list[tuple[str, int, str, str]] = []
    for rel in files:
        if rel in SKIP_PATHS or rel.startswith("venv/") or rel.startswith(".git/"):
            continue
        path = REPO_ROOT / rel
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if _is_binary(path):
            continue
        scanned += 1
        if args.verbose:
            print(f"  scanning {rel}")
        for lineno, name, excerpt in scan_file(path):
            findings.append((rel, lineno, name, excerpt))

    print(f"scan_repo: scanned {scanned} committable file(s) out of {len(files)} "
          "git-visible path(s)")
    if not findings:
        print("scan_repo: PASS - no phone numbers, UPI ids, emails or API keys found")
        return 0

    print(f"\nscan_repo: FAIL - {len(findings)} finding(s). Values are masked below.\n")
    for rel, lineno, name, excerpt in findings:
        print(f"  {rel}:{lineno}  [{name}]  {excerpt}")
    print("\nFix: remove the value, replace it with a placeholder, add a line "
          "marker `scan:allow` for documentation examples, or gitignore the file.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
