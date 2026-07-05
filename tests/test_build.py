from text2sql.data.build import dedupe, leakage_audit, length_stats, norm_sql
from text2sql.data.spider import Example


def ex(i: int, db: str, q: str, sql: str, split: str = "train") -> Example:
    return Example(f"{split}-{i:05d}", split, db, q, sql)


def test_norm_sql_ignores_spacing_case_and_semicolon():
    assert norm_sql("SELECT name ,  age FROM singer;") == norm_sql("select name, age from singer")
    assert norm_sql('SELECT * FROM t WHERE a = "x"') == norm_sql("SELECT * FROM t WHERE a = 'x'")


def test_dedupe_removes_only_exact_repeats():
    rows = [
        ex(0, "a", "How many singers?", "SELECT count(*) FROM singer"),
        ex(1, "a", "How many  singers ?", "select count(*) from singer"),  # exact repeat
        ex(2, "a", "Count the singers.", "SELECT count(*) FROM singer"),  # paraphrase, kept
        ex(3, "b", "How many singers?", "SELECT count(*) FROM singer"),  # other database, kept
        ex(4, "a", "How many singers?", "SELECT count(singer_id) FROM singer"),  # other SQL, kept
    ]
    kept, report = dedupe(rows)
    assert [e.id for e in kept] == ["train-00000", "train-00002", "train-00003", "train-00004"]
    assert report.exact_duplicates_removed == 1
    assert report.same_question_different_sql == 1
    assert report.same_sql_different_question == 1


def test_leakage_audit_flags_cross_split_pairs():
    train = [
        ex(0, "a", "How many singers?", "SELECT count(*) FROM singer"),
        ex(1, "a", "List all names.", "SELECT name FROM people"),
    ]
    dev = [
        ex(0, "z", "How many singers?", "SELECT count(*) FROM singer", "dev"),
        ex(1, "a", "Other question", "SELECT 1", "dev"),
    ]
    audit = leakage_audit(train, dev)
    assert audit["shared_databases"] == ["a"]
    assert audit["train_question_and_sql_equal_to_a_dev_pair"] == 1
    assert audit["removed_ids"] == ["train-00000"]


def test_length_stats():
    stats = length_stats(list(range(1, 101)))
    assert stats["min"] == 1 and stats["max"] == 100
    assert stats["p50"] == 51
    assert length_stats([]) == {}
