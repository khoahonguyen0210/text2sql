"""Score a saved prediction file with the pinned official Spider evaluator. No GPU needed."""

import argparse
import json
from pathlib import Path

from text2sql.data.spider import load_examples
from text2sql.evaluation.records import read_predictions
from text2sql.evaluation.score import score_predictions
from text2sql.manifest import read_json, sha256_file, utc_now, write_json
from text2sql.paths import DATA_DIR, SPIDER_DIR, relative

DB_DIRS = {"spider": SPIDER_DIR / "database", "testsuite": DATA_DIR / "testsuite" / "database"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="folder with predictions.jsonl")
    parser.add_argument("--db-set", choices=list(DB_DIRS), default="spider")
    parser.add_argument("--skip-official-cli", action="store_true",
                        help="per-example scoring only (faster, for quick checks)")
    parser.add_argument("--out", type=Path, help="output folder (default: RUN_DIR/score-DB_SET)")
    args = parser.parse_args()

    pred_file = args.run_dir / "predictions.jsonl"
    manifest_file = args.run_dir / "manifest.json"
    if not manifest_file.exists():  # runs saved in results/ by collect_results.py
        manifest_file = args.run_dir / "generation_manifest.json"
    manifest = read_json(manifest_file)
    preds = read_predictions(pred_file)
    examples = load_examples(manifest["split"])
    if manifest.get("subset", "full") != "full":
        examples = examples[: manifest["example_count"]]

    out_dir = args.out or args.run_dir / f"score-{args.db_set}"
    result = score_predictions(
        examples, preds, DB_DIRS[args.db_set], SPIDER_DIR / "tables.json", out_dir,
        db_set=args.db_set, run_official_cli=not args.skip_official_cli,
    )
    metrics = {
        "benchmark": "Spider 1.0",
        "run_id": manifest["run_id"],
        "subset": manifest.get("subset", "full"),
        "model": manifest["model"],
        "prompt": manifest["prompt"],
        "decoding": manifest["decoding"],
        "predictions_file": relative(pred_file),
        "predictions_sha256": sha256_file(pred_file),
        "scored_at": utc_now(),
        **result,
    }
    write_json(out_dir / "metrics.json", metrics)
    summary = {k: metrics[k] for k in ("run_id", "subset", "example_count", "scores", "correct")}
    print(json.dumps(summary | {"matches_cli": result["per_example_matches_official_cli"]}))


if __name__ == "__main__":
    main()
