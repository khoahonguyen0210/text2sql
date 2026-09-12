"""Label wrong predictions, compare two runs, and draw a sample for reading by hand.

If review.csv exists in the output folder, the hand labels are compared with the rule labels.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from text2sql.data.spider import load_examples
from text2sql.evaluation.errors import CATEGORIES, label_failure
from text2sql.manifest import utc_now, write_json
from text2sql.paths import RESULTS_DIR

REVIEW_FIELDS = ["id", "run", "db_id", "hardness", "question", "gold", "prediction",
                 "rule_label", "review_label", "note"]


def load_rows(score_dir: Path) -> list[dict]:
    text = (score_dir / "per_example.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def label_run(rows: list[dict], gold: dict[str, str]) -> list[dict]:
    for row in rows:
        row["labels"] = [] if row["exec"] else label_failure(
            row["sql"], gold[row["id"]], row["screen_status"], row["error"])
    return rows


def summarize(rows: list[dict]) -> dict:
    wrong = [r for r in rows if not r["exec"]]
    primary = Counter(r["labels"][0] for r in wrong)
    any_label = Counter(label for r in wrong for label in r["labels"])
    return {
        "examples": len(rows),
        "wrong": len(wrong),
        "main_label": {c: primary.get(c, 0) for c in CATEGORIES},
        "any_label": {c: any_label.get(c, 0) for c in CATEGORIES},
    }


def compare(a: list[dict], b: list[dict]) -> dict:
    pairs = list(zip(a, b, strict=True))
    fixed = [x["id"] for x, y in pairs if not x["exec"] and y["exec"]]
    broken = [x["id"] for x, y in pairs if x["exec"] and not y["exec"]]
    by_hardness: dict[str, dict] = {}
    for x, y in pairs:
        h = by_hardness.setdefault(x["hardness"], {"count": 0, "a": 0, "b": 0})
        h["count"] += 1
        h["a"] += x["exec"]
        h["b"] += y["exec"]
    return {
        "both_right": sum(1 for x, y in pairs if x["exec"] and y["exec"]),
        "both_wrong": sum(1 for x, y in pairs if not x["exec"] and not y["exec"]),
        "fixed": len(fixed),
        "broken": len(broken),
        "fixed_ids": fixed,
        "broken_ids": broken,
        "correct_by_hardness": by_hardness,
    }


def write_review_sample(path: Path, runs: dict[str, list[dict]], questions: dict[str, str],
                        gold: dict[str, str], size: int, seed: int) -> None:
    rng = random.Random(seed)
    rows = []
    for name, run in runs.items():
        wrong = [r for r in run if not r["exec"]]
        for r in rng.sample(wrong, min(size, len(wrong))):
            rows.append({"id": r["id"], "run": name, "db_id": r["db_id"],
                         "hardness": r["hardness"], "question": questions[r["id"]],
                         "gold": gold[r["id"]], "prediction": r["sql"],
                         "rule_label": r["labels"][0], "review_label": "", "note": ""})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def review_agreement(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["review_label"]]
    if not rows:
        return None
    same = sum(1 for r in rows if r["review_label"] == r["rule_label"])
    return {
        "reviewed": len(rows),
        "rule_label_matches_review": same,
        "agreement": round(same / len(rows), 3),
        "review_main_label": dict(Counter(r["review_label"] for r in rows).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "SCORE_DIR"),
                        required=True, type=str)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "error_analysis")
    parser.add_argument("--sample", type=int, default=25, help="wrong cases per run to read")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dev = load_examples("dev")
    gold = {e.id: e.query for e in dev}
    questions = {e.id: e.question for e in dev}
    runs = {name: label_run(load_rows(Path(d)), gold) for name, d in args.run}
    args.out.mkdir(parents=True, exist_ok=True)

    for name, rows in runs.items():
        with (args.out / f"labels_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                if not r["exec"]:
                    f.write(json.dumps({k: r[k] for k in ("id", "db_id", "hardness", "labels",
                                                          "screen_status", "sql")}) + "\n")

    names = list(runs)
    review_file = args.out / "review.csv"
    if not review_file.exists():
        write_review_sample(review_file, runs, questions, gold, args.sample, args.seed)
    summary = {
        "created_at": utc_now(),
        "metric": "execution match from the official evaluator, per example",
        "runs": {name: summarize(rows) for name, rows in runs.items()},
        "comparison": compare(runs[names[0]], runs[names[1]]) if len(names) == 2 else None,
        "compared": names,
        "review": review_agreement(review_file),
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps({n: s["main_label"] for n, s in summary["runs"].items()}, indent=2))
    if summary["comparison"]:
        print({k: summary["comparison"][k] for k in ("fixed", "broken", "both_wrong")})
    print("review:", summary["review"])


if __name__ == "__main__":
    main()
