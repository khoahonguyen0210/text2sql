import re

PLACEHOLDER_SQL = "SELECT"  # invalid on purpose: always scored as wrong

_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)(?:```|$)", re.IGNORECASE | re.DOTALL)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def _first_statement(text: str) -> tuple[str, bool]:
    """Cut text at the first semicolon outside quotes. Returns (sql, more_follows)."""
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == ";":
            rest = text[i + 1 :].strip()
            return text[:i], bool(rest)
    return text, False


def one_line(sql: str) -> str:
    """Join lines with single spaces, but keep text inside quotes unchanged."""
    out, quote, pending_space = [], None, False
    for ch in sql.strip():
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch.isspace():
            pending_space = True
            continue
        if pending_space and out:
            out.append(" ")
        pending_space = False
        out.append(ch)
        if ch in "'\"`":
            quote = ch
    return "".join(out).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def extract_sql(raw: str) -> tuple[str, str]:
    """Pull one SQL query out of model output. Returns (sql, status)."""
    text = (raw or "").strip()
    if not text:
        return "", "empty"
    status = "ok"
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
        status = "fenced"
    match = _SQL_START.search(text)
    if not match:
        return "", "no_sql"
    if match.start() > 0:
        status = "extracted"
    sql, more = _first_statement(text[match.start() :])
    sql = one_line(sql)
    if more:
        status = "multiple_statements"
    return sql, status
