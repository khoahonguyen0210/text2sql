import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from text2sql.data.spider import Example
from text2sql.evaluation import official
from text2sql.evaluation.extract import PLACEHOLDER_SQL
from text2sql.evaluation.records import Prediction, align, check_contract
from text2sql.sqlite_exec import execute_readonly

SCREEN_TIMEOUT_S = 60.0  # same value as TIMEOUT in exec_eval.py
SCREEN_ROW_CAP = 1_000_000
DB_SETS = {"spider": "EX", "testsuite": "TS"}


@dataclass
class ScreenResult:
    sql: str
    status: str  # ok, exec_error, empty, missing, denied, timeout, too_many_rows
    error: str = ""


def screen(sql: str, db_files: list[Path], timeout_s: float = SCREEN_TIMEOUT_S) -> ScreenResult:
    """Run a prediction read-only on every database file before the official evaluator sees it.

    Write attempts, time-outs and huge results become the placeholder, which is scored as wrong.
    """
    if not sql.strip():
        return ScreenResult(PLACEHOLDER_SQL, "empty")
    first_error = ""
    for path in db_files:
        result = execute_readonly(path, sql, timeout_s=timeout_s, max_rows=SCREEN_ROW_CAP)
        if result.status in ("denied", "timeout"):
            return ScreenResult(PLACEHOLDER_SQL, result.status, result.error)
        if result.truncated:
            return ScreenResult(PLACEHOLDER_SQL, "too_many_rows")
        if result.status == "error":
            first_error = result.error
            break  # fails the same way in the official evaluator
    status = "exec_error" if first_error else "ok"
    return ScreenResult(sql, status, first_error)


def db_files_for(db_dir: Path, db_id: str) -> list[Path]:
    folder = Path(db_dir) / db_id
    main = folder / f"{db_id}.sqlite"
    others = sorted(p for p in folder.glob("*.sqlite") if p != main)
    return [main, *others]


def score_predictions(
    examples: list[Example],
    preds: list[Prediction],
    db_dir: Path,
    table_file: Path,
    out_dir: Path,
    db_set: str = "spider",
    run_official_cli: bool = True,
    timeout_s: float = SCREEN_TIMEOUT_S,
) -> dict:
    """Score saved predictions. Writes official files, per-example rows and metrics."""
    if db_set not in DB_SETS:
        raise ValueError(f"unknown db set: {db_set}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    official.make_readonly(db_dir)
    contract = check_contract(examples, preds)

    screened: list[ScreenResult] = []
    for ex, pred in zip(examples, align(examples, preds), strict=True):
        if pred is None:
            screened.append(ScreenResult(PLACEHOLDER_SQL, "missing"))
        else:
            screened.append(screen(pred.sql, db_files_for(db_dir, ex.db_id), timeout_s))

    gold_file, pred_file = out_dir / "gold.txt", out_dir / "pred.txt"
    gold_file.write_text(
        "".join(f"{official.flat(e.query)}\t{e.db_id}\n" for e in examples), encoding="utf-8"
    )
    pred_file.write_text("".join(f"{official.flat(s.sql)}\n" for s in screened), encoding="utf-8")

    official_scores = None
    if run_official_cli:
        stdout = official.run_official(gold_file, pred_file, db_dir, table_file)
        (out_dir / "official_output.txt").write_text(stdout, encoding="utf-8")
        official_scores = official.parse_official(stdout)

    scorer = official.OfficialScorer(db_dir, table_file)
    rows = []
    aligned = align(examples, preds)
    for ex, pred, s in zip(examples, aligned, screened, strict=True):
        es = scorer.score(ex.query, ex.db_id, s.sql)
        rows.append({
            "id": ex.id,
            "db_id": ex.db_id,
            "hardness": es.hardness,
            "exec": es.exec,
            "exact": es.exact,
            "extraction_status": pred.extraction_status if pred else "missing",
            "screen_status": s.status,
            "error": s.error[:300],
            "sql": s.sql,
        })
    with (out_dir / "per_example.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_level = _by_level(rows)
    consistent = None
    if official_scores is not None:
        consistent = all(
            round(by_level[lvl][k], 3) == round(official_scores[lvl][k], 3)
            and by_level[lvl]["count"] == official_scores[lvl]["count"]
            for lvl in official.LEVELS
            for k in ("exec", "exact")
        )
    metric = DB_SETS[db_set]
    n = len(rows)
    return {
        "split": examples[0].split if examples else None,
        "example_count": n,
        "db_set": db_set,
        "headline_metric": metric,
        "scores": {metric: by_level["all"]["exec"], "EM": by_level["all"]["exact"]},
        "correct": {metric: sum(r["exec"] for r in rows), "EM": sum(r["exact"] for r in rows)},
        "by_hardness": by_level,
        "official_cli": official_scores,
        "per_example_matches_official_cli": consistent,
        "contract": asdict(contract) | {"ok": contract.ok},
        "extraction_status": dict(Counter(r["extraction_status"] for r in rows)),
        "screen_status": dict(Counter(r["screen_status"] for r in rows)),
        "evaluator": {
            "repo": official.EVALUATOR_REPO,
            "commit": official.EVALUATOR_COMMIT,
            "options": "--etype all (no --plug_value, no --keep_distinct)",
        },
        "scoring_seconds": round(time.perf_counter() - started, 1),
    }


def _by_level(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for level in official.LEVELS:
        subset = rows if level == "all" else [r for r in rows if r["hardness"] == level]
        n = len(subset)
        result[level] = {
            "count": n,
            "exec": sum(r["exec"] for r in subset) / n if n else 0.0,
            "exact": sum(r["exact"] for r in subset) / n if n else 0.0,
        }
    return result
