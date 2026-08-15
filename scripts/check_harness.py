"""Harness check, not a model score: gold SQL scored as predictions must get 100%.

Also checks that scoring leaves every database file unchanged.
"""

import argparse
import json

from text2sql.data.spider import load_examples
from text2sql.evaluation.records import Prediction
from text2sql.evaluation.score import score_predictions
from text2sql.manifest import sha256_file, utc_now, write_json
from text2sql.paths import ARTIFACTS_DIR, DATA_DIR, RESULTS_DIR, SPIDER_DIR

DB_DIRS = {"spider": SPIDER_DIR / "database", "testsuite": DATA_DIR / "testsuite" / "database"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-set", choices=list(DB_DIRS), nargs="+", default=list(DB_DIRS))
    args = parser.parse_args()

    dev = load_examples("dev")
    preds = [Prediction(e.id, e.db_id, e.query, e.query, "ok", "harness-gold", "gold", "none")
             for e in dev]
    report = {"checked_at": utc_now(), "what": "dev gold SQL scored as predictions"}
    for db_set in args.db_set:
        db_dir = DB_DIRS[db_set]
        dev_dbs = sorted({e.db_id for e in dev})
        files = [p for d in dev_dbs for p in sorted((db_dir / d).glob("*.sqlite"))]
        before = {p.name: sha256_file(p) for p in files}
        result = score_predictions(dev, preds, db_dir, SPIDER_DIR / "tables.json",
                                   ARTIFACTS_DIR / "harness" / db_set, db_set=db_set)
        after = {p.name: sha256_file(p) for p in files}
        report[db_set] = {
            "example_count": result["example_count"],
            "scores": result["scores"],
            "per_example_matches_official_cli": result["per_example_matches_official_cli"],
            "database_files": len(files),
            "database_files_unchanged": before == after,
            "scoring_seconds": result["scoring_seconds"],
        }
        print(db_set, json.dumps(report[db_set]))
    write_json(RESULTS_DIR / "harness_check.json", report)
    ok = all(all(v == 1.0 for v in report[s]["scores"].values())
             and report[s]["database_files_unchanged"] for s in args.db_set)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
