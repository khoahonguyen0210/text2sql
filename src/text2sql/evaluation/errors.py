"""Rule-based error labels. They are a first sort, checked against a hand-read sample."""

from collections import Counter

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

CATEGORIES = (
    "output_format",
    "syntax_dialect",
    "timeout_resource",
    "table_selection",
    "column_linking",
    "join_path",
    "aggregation_grouping",
    "filter_value",
    "order_limit",
    "nested_query",
    "set_operation",
    "other",
)


def _parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except (SqlglotError, ValueError):
        return None


def _names(tree: exp.Expression, kind: type) -> Counter:
    return Counter(n.name.lower() for n in tree.find_all(kind) if n.name)


def _features(tree: exp.Expression) -> dict:
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    projections = select.expressions if select else []
    wheres = [w.this for w in tree.find_all(exp.Where)]
    return {
        "tables": set(_names(tree, exp.Table)),
        "select_cols": sorted(c.name.lower() for p in projections for c in p.find_all(exp.Column)),
        "joins": len(list(tree.find_all(exp.Join))),
        "aggs": sorted(type(a).__name__ for a in tree.find_all(exp.AggFunc)),
        "group": sorted(c.name.lower() for g in tree.find_all(exp.Group)
                        for c in g.find_all(exp.Column)),
        "having": len(list(tree.find_all(exp.Having))),
        "where_cols": sorted(c.name.lower() for w in wheres for c in w.find_all(exp.Column)),
        "where_values": sorted(str(v.this).lower() for w in wheres
                               for v in w.find_all(exp.Literal)),
        "where_ops": sorted(type(o).__name__ for w in wheres for o in w.find_all(exp.Predicate)),
        "order": [(sorted(c.name.lower() for c in o.find_all(exp.Column)),
                   bool(o.args.get("desc"))) for o in tree.find_all(exp.Ordered)],
        "limit": [lim.sql().lower() for lim in tree.find_all(exp.Limit)],
        "selects": len(list(tree.find_all(exp.Select))),
        "set_ops": sorted(type(s).__name__ for s in tree.find_all(exp.SetOperation)),
    }


def label_failure(pred_sql: str, gold_sql: str, screen_status: str, error: str) -> list[str]:
    """Labels for one wrong prediction. The first label is the main one."""
    if screen_status in ("empty", "missing"):
        return ["output_format"]
    if screen_status in ("timeout", "too_many_rows"):
        return ["timeout_resource"]
    if screen_status == "exec_error":
        message = error.lower()
        if "no such table" in message:
            return ["table_selection"]
        if "no such column" in message or "ambiguous column" in message:
            return ["column_linking"]
        return ["syntax_dialect"]
    pred, gold = _parse(pred_sql), _parse(gold_sql)
    if pred is None or gold is None:
        return ["syntax_dialect"]
    p, g = _features(pred), _features(gold)
    labels = []
    if p["set_ops"] != g["set_ops"]:
        labels.append("set_operation")
    if p["selects"] != g["selects"] and not p["set_ops"] and not g["set_ops"]:
        labels.append("nested_query")
    if p["tables"] != g["tables"]:
        labels.append("table_selection")
    if p["joins"] != g["joins"] and "table_selection" not in labels:
        labels.append("join_path")
    if p["aggs"] != g["aggs"] or p["group"] != g["group"] or p["having"] != g["having"]:
        labels.append("aggregation_grouping")
    if (p["where_cols"], p["where_ops"], p["where_values"]) != (
            g["where_cols"], g["where_ops"], g["where_values"]):
        labels.append("filter_value")
    if p["order"] != g["order"] or p["limit"] != g["limit"]:
        labels.append("order_limit")
    if p["select_cols"] != g["select_cols"]:
        labels.append("column_linking")
    return labels or ["other"]
