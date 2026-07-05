import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from text2sql.data.spider import Example
from text2sql.evaluation.extract import one_line

TRANSFORM_VERSION = "1"


def norm_question(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def norm_sql(sql: str) -> str:
    """Lower case, single spaces, no space before commas or after brackets, no final ';'."""
    text = one_line(sql).rstrip(";").strip().lower()
    text = re.sub(r"\s*([(),])\s*", r"\1", text)
    return text.replace('"', "'")


@dataclass
class DedupeReport:
    before: int
    after: int
    exact_duplicates_removed: int
    same_question_different_sql: int
    same_sql_different_question: int


def dedupe(examples: list[Example]) -> tuple[list[Example], DedupeReport]:
    """Remove exact repeats of (db_id, question, SQL). Keep paraphrases and repeated SQL shapes."""
    seen: set[tuple[str, str, str]] = set()
    kept: list[Example] = []
    for ex in examples:
        key = (ex.db_id, norm_question(ex.question), norm_sql(ex.query))
        if key not in seen:
            seen.add(key)
            kept.append(ex)

    sql_per_question: dict[tuple[str, str], set[str]] = defaultdict(set)
    questions_per_sql: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ex in kept:
        sql_per_question[(ex.db_id, norm_question(ex.question))].add(norm_sql(ex.query))
        questions_per_sql[(ex.db_id, norm_sql(ex.query))].add(norm_question(ex.question))
    return kept, DedupeReport(
        before=len(examples),
        after=len(kept),
        exact_duplicates_removed=len(examples) - len(kept),
        same_question_different_sql=sum(1 for v in sql_per_question.values() if len(v) > 1),
        same_sql_different_question=sum(1 for v in questions_per_sql.values() if len(v) > 1),
    )


def leakage_audit(train: list[Example], dev: list[Example]) -> dict:
    """Compare train against dev. Spider keeps databases apart, so overlaps should be rare."""
    train_dbs, dev_dbs = {e.db_id for e in train}, {e.db_id for e in dev}
    dev_sql = {norm_sql(e.query) for e in dev}
    dev_q = {norm_question(e.question) for e in dev}
    dev_pair = {(norm_question(e.question), norm_sql(e.query)) for e in dev}
    sql_hits = [e for e in train if norm_sql(e.query) in dev_sql]
    q_hits = [e for e in train if norm_question(e.question) in dev_q]
    pair_hits = [e for e in train if (norm_question(e.question), norm_sql(e.query)) in dev_pair]
    return {
        "shared_databases": sorted(train_dbs & dev_dbs),
        "train_sql_equal_to_a_dev_sql": len(sql_hits),
        "train_question_equal_to_a_dev_question": len(q_hits),
        "train_question_and_sql_equal_to_a_dev_pair": len(pair_hits),
        "examples_sql_overlap": [e.query for e in sql_hits[:10]],
        "examples_question_overlap": [e.question for e in q_hits[:10]],
        "removed_ids": sorted(e.id for e in pair_hits),
    }


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {}
    ordered = sorted(lengths)

    def pct(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
    }


def per_db_counts(examples: list[Example]) -> dict[str, int]:
    return dict(sorted(Counter(e.db_id for e in examples).items()))
