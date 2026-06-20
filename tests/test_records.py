from text2sql.data.spider import Example
from text2sql.evaluation.records import (
    Prediction,
    align,
    append_prediction,
    check_contract,
    read_predictions,
)


def example(i: int, db_id: str = "toy") -> Example:
    return Example(f"dev-{i:05d}", "dev", db_id, f"q{i}", "SELECT 1")


def pred(i: int, db_id: str = "toy", sql: str = "SELECT 1") -> Prediction:
    return Prediction(f"dev-{i:05d}", db_id, sql, sql, "ok", "run", "model", "v1")


def test_round_trip(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_prediction(path, pred(0))
    append_prediction(path, pred(1, sql="SELECT 'ä'"))
    assert read_predictions(path) == [pred(0), pred(1, sql="SELECT 'ä'")]


def test_contract_finds_problems():
    examples = [example(0), example(1), example(2)]
    preds = [pred(0), pred(0), pred(2, db_id="other"), pred(7)]
    report = check_contract(examples, preds)
    assert not report.ok
    assert report.missing == ["dev-00001"]
    assert report.duplicates == ["dev-00000"]
    assert report.unknown == ["dev-00007"]
    assert report.db_mismatch == ["dev-00002"]


def test_align_keeps_example_order():
    examples = [example(0), example(1), example(2)]
    aligned = align(examples, [pred(2), pred(0)])
    assert [p.id if p else None for p in aligned] == ["dev-00000", None, "dev-00002"]
