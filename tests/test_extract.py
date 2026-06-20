import pytest

from text2sql.evaluation.extract import extract_sql, one_line


@pytest.mark.parametrize(
    ("raw", "sql", "status"),
    [
        ("SELECT name FROM singer", "SELECT name FROM singer", "ok"),
        ("SELECT name FROM singer;", "SELECT name FROM singer", "ok"),
        ("", "", "empty"),
        ("   \n ", "", "empty"),
        ("I am not sure.", "", "no_sql"),
        ("```sql\nSELECT name\nFROM singer\n```", "SELECT name FROM singer", "fenced"),
        ("```\nSELECT 1\n```", "SELECT 1", "fenced"),
        ("```sql\nSELECT name FROM singer", "SELECT name FROM singer", "fenced"),
        ("Here is the query: SELECT name FROM singer", "SELECT name FROM singer", "extracted"),
        ("SELECT 1; DROP TABLE singer;", "SELECT 1", "multiple_statements"),
        ("WITH a AS (SELECT 1) SELECT * FROM a", "WITH a AS (SELECT 1) SELECT * FROM a", "ok"),
    ],
)
def test_extract(raw, sql, status):
    assert extract_sql(raw) == (sql, status)


def test_semicolon_inside_string_is_kept():
    sql, status = extract_sql("SELECT * FROM t WHERE name = 'a;b'")
    assert sql == "SELECT * FROM t WHERE name = 'a;b'"
    assert status == "ok"


def test_one_line_keeps_spaces_inside_quotes():
    assert one_line("SELECT  *\nFROM t\tWHERE x = 'New  York'") == (
        "SELECT * FROM t WHERE x = 'New  York'"
    )
