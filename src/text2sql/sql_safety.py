import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_SQL_CHARS = 4000

_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Pragma, exp.Attach, exp.Detach, exp.Command, exp.Transaction,
    exp.Commit, exp.Rollback, exp.Copy, exp.Set, exp.Analyze, exp.LoadData, exp.Into,
)
_FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


@dataclass(frozen=True)
class SafetyResult:
    ok: bool
    reason: str = ""


def check_sql(sql: str) -> SafetyResult:
    """Parser check: allow exactly one read-only SELECT (or UNION/INTERSECT/EXCEPT)."""
    text = sql.strip()
    if not text:
        return SafetyResult(False, "empty")
    if len(text) > MAX_SQL_CHARS:
        return SafetyResult(False, "too_long")
    try:
        statements = [s for s in sqlglot.parse(text, read="sqlite") if s is not None]
    except SqlglotError:
        return SafetyResult(False, "parse_error")
    if len(statements) != 1:
        return SafetyResult(False, "not_one_statement")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.SetOperation):
        return SafetyResult(False, "not_select")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return SafetyResult(False, "forbidden_statement")
        if isinstance(node, exp.Func):
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if name.lower() in _FORBIDDEN_FUNCTIONS:
                return SafetyResult(False, "forbidden_function")
    return SafetyResult(True)
