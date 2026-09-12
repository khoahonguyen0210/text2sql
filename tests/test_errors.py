import pytest

from text2sql.evaluation.errors import label_failure

GOLD = "SELECT name FROM singer WHERE age > 40 ORDER BY age DESC"


@pytest.mark.parametrize(
    ("pred", "status", "error", "main"),
    [
        ("", "empty", "", "output_format"),
        ("SELECT", "timeout", "", "timeout_resource"),
        ("SELECT x FROM nowhere", "exec_error", "no such table: nowhere", "table_selection"),
        ("SELECT nope FROM singer", "exec_error", "no such column: nope", "column_linking"),
        ("SELECT name FROM singer WHERE age > 40 ORDER BY age", "ok", "", "order_limit"),
        ("SELECT name FROM singer WHERE age > 30 ORDER BY age DESC", "ok", "", "filter_value"),
        ("SELECT country FROM singer WHERE age > 40 ORDER BY age DESC", "ok", "",
         "column_linking"),
        ("SELECT name FROM concert WHERE age > 40 ORDER BY age DESC", "ok", "",
         "table_selection"),
        ("SELECT count(name) FROM singer WHERE age > 40 ORDER BY age DESC", "ok", "",
         "aggregation_grouping"),
        ("SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer) ORDER BY age DESC",
         "ok", "", "nested_query"),
        ("SELECT name FROM singer WHERE age > 40 UNION SELECT name FROM singer", "ok", "",
         "set_operation"),
        ("SELECT name FROM singer WHERE age > 40 ORDER BY age DESC", "ok", "", "other"),
    ],
)
def test_main_label(pred, status, error, main):
    assert label_failure(pred, GOLD, status, error)[0] == main


def test_join_path():
    gold = "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id"
    pred = ("SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id "
            "JOIN concert AS T3 ON T3.id = T2.id")
    assert "join_path" in label_failure(pred, gold, "ok", "")
