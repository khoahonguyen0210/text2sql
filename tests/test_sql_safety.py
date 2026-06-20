import pytest

from text2sql.sql_safety import check_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM singer",
        "select count(*) from singer where country = 'France'",
        "SELECT name FROM singer UNION SELECT name FROM singer",
        "SELECT name FROM singer EXCEPT SELECT name FROM singer WHERE age > 40",
        "WITH old AS (SELECT * FROM singer WHERE age > 40) SELECT name FROM old",
        "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id",
    ],
)
def test_allows_read_only_queries(sql):
    assert check_sql(sql).ok


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("", "empty"),
        ("SELECT 1; DROP TABLE singer", "not_one_statement"),
        ("DROP TABLE singer", "not_select"),
        ("DELETE FROM singer", "not_select"),
        ("INSERT INTO singer VALUES (9, 'x', 'y', 1)", "not_select"),
        ("UPDATE singer SET age = 1", "not_select"),
        ("PRAGMA table_info(singer)", "not_select"),
        ("ATTACH DATABASE 'other.db' AS other", "not_select"),
        ("VACUUM", "not_select"),
        ("SELECT load_extension('evil')", "forbidden_function"),
        ("SELECT readfile('secret.txt')", "forbidden_function"),
        ("SELECT * FROM (", "parse_error"),
        ("SELECT " + "1 + " * 2000 + "1", "too_long"),
    ],
)
def test_rejects_unsafe_queries(sql, reason):
    result = check_sql(sql)
    assert not result.ok
    assert result.reason == reason
