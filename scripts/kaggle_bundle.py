"""Pack code and processed data for Kaggle GPU runs, with one notebook folder per plan.

The bundle holds dev questions without gold SQL. The dataset and notebooks are private.
"""

import argparse
import json
import shutil
from pathlib import Path

from text2sql.manifest import git_state, write_json
from text2sql.paths import PROCESSED_DIR, ROOT, SPIDER_DIR

BUNDLE_DIR = ROOT / "kaggle" / "bundle"
DATASET_SLUG = "text2sql-qlora-bundle"
OUT = "/kaggle/working/artifacts/runs"


def train_and_generate(config: str, run_id: str) -> list[list[str]]:
    return [
        ["scripts/train_qlora.py", "--config", config, "--run-id", run_id],
        ["scripts/generate.py", "--config", "configs/base.toml", "--adapter",
         f"{OUT}/{run_id}/adapter", "--run-id", f"{run_id}-dev"],
    ]


PLANS = {
    "smoke": [
        ["scripts/train_qlora.py", "--config", "configs/qlora.toml", "--max-steps", "10",
         "--limit", "64", "--run-id", "smoke-train"],
        ["scripts/generate.py", "--config", "configs/base.toml", "--limit", "16",
         "--run-id", "smoke-base"],
        ["scripts/generate.py", "--config", "configs/base.toml", "--limit", "16",
         "--adapter", f"{OUT}/smoke-train/adapter", "--run-id", "smoke-tuned"],
    ],
    "base": [
        ["scripts/generate.py", "--config", "configs/base.toml", "--run-id", "base-dev"],
        ["scripts/generate.py", "--config", "configs/base_nofk.toml",
         "--run-id", "base-nofk-dev"],
    ],
    "qlora": train_and_generate("configs/qlora.toml", "qlora-r16"),
    "qlora-r8": train_and_generate("configs/qlora_r8.toml", "qlora-r8"),
}


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        skip = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
        shutil.copytree(src, dst, ignore=skip)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="your Kaggle user name")
    args = parser.parse_args()

    questions = PROCESSED_DIR / "dev_questions.jsonl"
    for line in questions.read_text(encoding="utf-8").splitlines():
        if "query" in json.loads(line):
            raise SystemExit("dev_questions.jsonl must not hold gold SQL")

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    data_dir = BUNDLE_DIR / "dataset"
    for name in ("src", "scripts", "configs"):
        copy(ROOT / name, data_dir / name)
    for name in ("train.jsonl", "dev_questions.jsonl", "manifest.json"):
        copy(PROCESSED_DIR / name, data_dir / "data" / "processed" / name)
    for name in ("tables.json", "SOURCE.json"):
        if (SPIDER_DIR / name).exists():
            copy(SPIDER_DIR / name, data_dir / "data" / "spider" / name)
    git = git_state()
    write_json(data_dir / "bundle.json",
               {"commit": git["commit"], "dirty": git["dirty"], "plans": PLANS})
    write_json(data_dir / "dataset-metadata.json", {
        "title": "text2sql qlora bundle",
        "id": f"{args.user}/{DATASET_SLUG}",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    })

    template = (ROOT / "kaggle" / "run.py").read_text(encoding="utf-8")
    for plan in PLANS:
        kernel_dir = BUNDLE_DIR / "kernels" / plan
        kernel_dir.mkdir(parents=True)
        code = template.replace('PLAN = "smoke"', f'PLAN = "{plan}"', 1)
        (kernel_dir / "run.py").write_text(code, encoding="utf-8")
        write_json(kernel_dir / "kernel-metadata.json", {
            "id": f"{args.user}/text2sql-{plan}",
            "title": f"text2sql {plan}",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{args.user}/{DATASET_SLUG}"],
        })
    print(f"bundle ready in {BUNDLE_DIR.relative_to(ROOT).as_posix()}: {', '.join(PLANS)}")


if __name__ == "__main__":
    main()
