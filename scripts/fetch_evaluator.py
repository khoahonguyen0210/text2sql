"""Clone the official Spider evaluator at a pinned commit and fetch its NLTK data."""

import subprocess
import sys

import nltk

from text2sql.evaluation.official import EVALUATOR_COMMIT, EVALUATOR_REPO
from text2sql.paths import EVALUATOR_DIR, NLTK_DIR


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def main() -> None:
    if not (EVALUATOR_DIR / ".git").exists():
        EVALUATOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--quiet", EVALUATOR_REPO, str(EVALUATOR_DIR))
    git("-C", str(EVALUATOR_DIR), "fetch", "--quiet", "origin")
    git("-C", str(EVALUATOR_DIR), "checkout", "--quiet", EVALUATOR_COMMIT)
    head = git("-C", str(EVALUATOR_DIR), "rev-parse", "HEAD")
    if head != EVALUATOR_COMMIT:
        sys.exit(f"evaluator is at {head}, expected {EVALUATOR_COMMIT}")
    if not nltk.download("punkt_tab", download_dir=str(NLTK_DIR), quiet=True):
        sys.exit("could not download NLTK punkt_tab")
    print(f"evaluator ready at commit {head[:12]}")


if __name__ == "__main__":
    main()
