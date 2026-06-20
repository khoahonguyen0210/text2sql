import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_ROWS = 1000

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


@dataclass
class ExecResult:
    status: str  # ok, error, timeout, denied
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    truncated: bool = False
    error: str = ""
    elapsed_ms: float = 0.0


def _authorizer(action: int, arg1, arg2, db_name, trigger) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a database file read-only. Writes, ATTACH and PRAGMA are blocked."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    conn.text_factory = lambda b: b.decode(errors="replace")
    conn.set_authorizer(_authorizer)
    return conn


def execute_readonly(
    db_path: Path,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> ExecResult:
    """Run one query with a time limit and a row limit. Never raises for bad SQL."""
    start = time.perf_counter()
    deadline = start + timeout_s
    timed_out = False

    def check_time() -> int:
        nonlocal timed_out
        if time.perf_counter() > deadline:
            timed_out = True
            return 1  # non-zero aborts the query
        return 0

    try:
        conn = connect_readonly(db_path)
    except sqlite3.Error as e:
        return ExecResult("error", error=f"open failed: {e}")
    try:
        conn.set_progress_handler(check_time, 10_000)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
        truncated = len(rows) > max_rows
        return ExecResult(
            "ok", rows[:max_rows], columns, truncated, elapsed_ms=_ms(start)
        )
    except sqlite3.DatabaseError as e:
        if timed_out:
            return ExecResult("timeout", error="time limit reached", elapsed_ms=_ms(start))
        status = "denied" if "not authorized" in str(e) else "error"
        return ExecResult(status, error=str(e), elapsed_ms=_ms(start))
    except (sqlite3.Error, ValueError, OverflowError) as e:
        return ExecResult("error", error=str(e), elapsed_ms=_ms(start))
    finally:
        conn.close()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
