import pytest

from text2sql.manifest import sha256_file
from text2sql.sqlite_exec import execute_readonly


def test_reads_rows(toy_db):
    sql = "SELECT name FROM singer WHERE country = 'France' ORDER BY name"
    result = execute_readonly(toy_db, sql)
    assert result.status == "ok"
    assert result.rows == [("Justin Brown",), ("Rose White",)]
    assert result.columns == ["name"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM singer",
        "DROP TABLE singer",
        "INSERT INTO singer VALUES (9, 'x', 'y', 1)",
        "UPDATE singer SET age = 0",
        "CREATE TABLE x (a)",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA writable_schema = ON",
    ],
)
def test_writes_are_blocked_and_file_is_unchanged(toy_db, sql):
    before = sha256_file(toy_db)
    result = execute_readonly(toy_db, sql)
    assert result.status in ("denied", "error")
    assert sha256_file(toy_db) == before
    assert execute_readonly(toy_db, "SELECT count(*) FROM singer").rows == [(4,)]


def test_multiple_statements_fail(toy_db):
    result = execute_readonly(toy_db, "SELECT 1; DELETE FROM singer")
    assert result.status == "error"


def test_time_limit_stops_endless_query(toy_db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    result = execute_readonly(toy_db, sql, timeout_s=0.5)
    assert result.status == "timeout"
    assert result.elapsed_ms < 5000


def test_row_limit(toy_db):
    result = execute_readonly(toy_db, "SELECT * FROM singer", max_rows=2)
    assert result.truncated
    assert len(result.rows) == 2


def test_bad_sql_returns_error(toy_db):
    result = execute_readonly(toy_db, "SELECT nope FROM singer")
    assert result.status == "error"
    assert "nope" in result.error
