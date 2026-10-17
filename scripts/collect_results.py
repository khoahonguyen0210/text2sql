"""Copy the small files of finished runs into results/ and write one result table.

Model weights are never copied.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from text2sql.manifest import read_json, write_json
from text2sql.paths import RESULTS_DIR

FIELDS = ["name", "benchmark_version", "split", "example_count", "metric", "score", "correct",
          "model_id", "model_revision", "adapter", "prompt_version", "prompt_hash", "decoding",
          "evaluator_revision", "db_set", "run_id"]


def rows_for(name: str, run_dir: Path) -> list[dict]:
    rows = []
    for score_dir in sorted(run_dir.glob("score-*")):
        m = read_json(score_dir / "metrics.json")
        for metric in (m["headline_metric"], "EM"):
            rows.append({
                "name": name,
                "benchmark_version": "Spider 1.0",
                "split": m["split"],
                "example_count": m["example_count"],
                "metric": metric,
                "score": round(m["scores"][metric], 4),
                "correct": m["correct"][metric],
                "model_id": m["model"]["id"],
                "model_revision": m["model"]["revision"],
                "adapter": m["model"]["adapter"] or "",
                "prompt_version": m["prompt"]["version"],
                "prompt_hash": m["prompt"]["hash"],
                "decoding": json.dumps(m["decoding"], sort_keys=True),
                "evaluator_revision": m["evaluator"]["commit"],
                "db_set": m["db_set"],
                "run_id": m["run_id"],
            })
    return rows


def add_commit(path: Path, commit: str | None) -> None:
    """Kaggle has no git, so the commit comes from the bundle the run used."""
    data = read_json(path)
    if commit and not data["environment"]["git"]["commit"]:
        data["code_commit"] = commit
        write_json(path, data)


def copy_run(name: str, run_dir: Path, train_dir: Path | None, out: Path,
             commit: str | None) -> None:
    dest = out / "runs" / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "predictions.jsonl", dest / "predictions.jsonl")
    shutil.copy2(run_dir / "manifest.json", dest / "generation_manifest.json")
    add_commit(dest / "generation_manifest.json", commit)
    for score_dir in sorted(run_dir.glob("score-*")):
        tag = score_dir.name.removeprefix("score-")
        shutil.copy2(score_dir / "metrics.json", dest / f"metrics_{tag}.json")
        shutil.copy2(score_dir / "per_example.jsonl", dest / f"per_example_{tag}.jsonl")
    if train_dir:
        shutil.copy2(train_dir / "manifest.json", dest / "train_manifest.json")
        add_commit(dest / "train_manifest.json", commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "RUN_DIR"),
                        required=True)
    parser.add_argument("--train", nargs=2, action="append", metavar=("NAME", "TRAIN_DIR"),
                        default=[])
    parser.add_argument("--code-commit", help="commit of the Kaggle bundle the runs used")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    train_dirs = {name: Path(d) for name, d in args.train}
    rows = []
    for name, run_dir in args.run:
        copy_run(name, Path(run_dir), train_dirs.get(name), args.out, args.code_commit)
        rows.extend(rows_for(name, Path(run_dir)))
    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out / "results.json", {"rows": rows})
    for row in rows:
        print(f"{row['name']:<12} {row['db_set']:<10} {row['metric']:<3} "
              f"{row['score']:.3f} ({row['correct']}/{row['example_count']})")


if __name__ == "__main__":
    main()
