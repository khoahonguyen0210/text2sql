from tests.conftest import needs_evaluator
from text2sql.data.spider import Example
from text2sql.evaluation.records import Prediction
from text2sql.evaluation.score import score_predictions, screen
from text2sql.manifest import sha256_file

GOLD = [
    "SELECT count(*) FROM singer",
    "SELECT name FROM singer WHERE country = 'France'",
    "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id "
    "WHERE T2.year = 2015",
    "SELECT name FROM singer ORDER BY age DESC LIMIT 1",
    "SELECT avg(age) FROM singer",
]


def examples() -> list[Example]:
    return [Example(f"dev-{i:05d}", "dev", "toy", f"q{i}", q) for i, q in enumerate(GOLD)]


def preds(sqls: list[str]) -> list[Prediction]:
    return [
        Prediction(f"dev-{i:05d}", "toy", s, s, "ok", "run", "model", "v1")
        for i, s in enumerate(sqls)
    ]


def test_screen_blocks_writes(toy_db):
    before = sha256_file(toy_db)
    result = screen("DELETE FROM singer", [toy_db])
    assert result.status == "denied"
    assert result.sql == "SELECT"
    assert sha256_file(toy_db) == before


def test_screen_stops_endless_query(toy_db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    assert screen(sql, [toy_db], timeout_s=0.5).status == "timeout"


@needs_evaluator
def test_gold_as_prediction_scores_full_marks(toy_root, tmp_path):
    result = score_predictions(
        examples(), preds(GOLD), toy_root / "database", toy_root / "tables.json", tmp_path / "out"
    )
    assert result["scores"] == {"EX": 1.0, "EM": 1.0}
    assert result["per_example_matches_official_cli"] is True


@needs_evaluator
def test_bad_predictions_are_counted_not_dropped(toy_root, tmp_path):
    db = toy_root / "database" / "toy" / "toy.sqlite"
    before = sha256_file(db)
    sqls = [
        "SELECT count(*) FROM singer",  # right
        "",  # empty
        "DELETE FROM singer",  # write attempt
        "SELECT name FROM nowhere",  # execution error
    ]
    result = score_predictions(
        examples(), preds(sqls), toy_root / "database", toy_root / "tables.json", tmp_path / "out"
    )
    assert result["example_count"] == 5
    assert result["correct"]["EX"] == 1
    assert result["screen_status"] == {
        "ok": 1, "empty": 1, "denied": 1, "exec_error": 1, "missing": 1,
    }
    assert result["contract"]["missing"] == ["dev-00004"]
    assert result["per_example_matches_official_cli"] is True
    assert sha256_file(db) == before
