"""Wrapper around the pinned official Spider evaluator (taoyds/test-suite-sql-eval)."""

import asyncio
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from text2sql.paths import EVALUATOR_DIR, NLTK_DIR

EVALUATOR_REPO = "https://github.com/taoyds/test-suite-sql-eval"
EVALUATOR_COMMIT = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"
LEVELS = ("easy", "medium", "hard", "extra", "all")

_EMPTY_SQL = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}


class OneLoop:
    """Stands in for asyncio inside exec_eval.py, with one event loop for all queries.

    exec_eval.py calls asyncio.run() per query and database. On Windows each new loop opens
    two local sockets, so a test-suite run (about 70,000 calls) runs out of ports and hangs.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()

    def __getattr__(self, name: str):
        return getattr(asyncio, name)

    def run(self, coro):
        return self._loop.run_until_complete(coro)


def patch_exec_eval() -> None:
    import exec_eval  # type: ignore[import-not-found]

    if not isinstance(exec_eval.asyncio, OneLoop):
        exec_eval.asyncio = OneLoop()


def flat(sql: str) -> str:
    """The official files hold one query per line, split on tabs."""
    return re.sub(r"[\t\r\n]+", " ", sql).strip()


def make_readonly(db_root: Path) -> int:
    """Mark every SQLite file read-only, as a second guard under the screening step."""
    count = 0
    for path in Path(db_root).rglob("*.sqlite"):
        os.chmod(path, stat.S_IREAD)
        count += 1
    return count


def evaluator_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["NLTK_DATA"] = str(NLTK_DIR)
    return env


def run_official(
    gold_file: Path, pred_file: Path, db_dir: Path, table_file: Path, timeout_s: int = 7200
) -> str:
    """Run the unmodified evaluation.py and return what it prints."""
    cmd = [
        sys.executable,
        "-m",
        "text2sql.evaluation.official_cli",
        "--gold", str(Path(gold_file).resolve()),
        "--pred", str(Path(pred_file).resolve()),
        "--db", str(Path(db_dir).resolve()),
        "--table", str(Path(table_file).resolve()),
        "--etype", "all",
    ]
    done = subprocess.run(
        cmd,
        cwd=EVALUATOR_DIR,
        env=evaluator_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_s,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"official evaluator failed:\n{done.stderr[-2000:]}")
    return done.stdout


def parse_official(stdout: str) -> dict[str, dict[str, float]]:
    """Read the count, execution and exact-match rows from the printed table."""
    rows: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        for key in ("count", "execution", "exact match"):
            if line.startswith(key + " "):
                rows.setdefault(key, line[len(key):].split())
    result: dict[str, dict[str, float]] = {}
    for i, level in enumerate(LEVELS):
        result[level] = {
            "count": int(rows["count"][i]),
            "exec": float(rows["execution"][i]),
            "exact": float(rows["exact match"][i]),
        }
    return result


@dataclass
class ExampleScore:
    hardness: str
    exec: int
    exact: int


class OfficialScorer:
    """Scores one example at a time with the official functions, for per-example results."""

    def __init__(self, db_dir: Path, table_file: Path):
        if str(EVALUATOR_DIR) not in sys.path:
            sys.path.insert(0, str(EVALUATOR_DIR))
        import nltk

        if str(NLTK_DIR) not in nltk.data.path:
            nltk.data.path.insert(0, str(NLTK_DIR))
        import evaluation as ev  # type: ignore[import-not-found]

        patch_exec_eval()
        self.ev = ev
        self.db_dir = Path(db_dir)
        self.kmaps = ev.build_foreign_key_map_from_json(str(table_file))
        self.evaluator = ev.Evaluator()

    def score(self, gold: str, db_id: str, pred: str) -> ExampleScore:
        # Mirrors the loop body of evaluate() in evaluation.py.
        ev = self.ev
        p_str = pred.strip().replace("value", "1")
        g_str = gold.strip()
        db = os.path.join(self.db_dir, db_id, db_id + ".sqlite")
        schema = ev.Schema(ev.get_schema(db))
        g_sql = ev.get_sql(schema, g_str)
        hardness = self.evaluator.eval_hardness(g_sql)
        try:
            p_sql = ev.get_sql(schema, p_str)
        except Exception:
            p_sql = _empty_sql()
        exec_score = ev.eval_exec_match(
            db=db, p_str=p_str, g_str=g_str, plug_value=False,
            keep_distinct=False, progress_bar_for_each_datapoint=False,
        )
        kmap = self.kmaps[db_id]
        g_units = ev.build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = ev.rebuild_sql_col(g_units, ev.rebuild_sql_val(g_sql), kmap)
        p_units = ev.build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = ev.rebuild_sql_col(p_units, ev.rebuild_sql_val(p_sql), kmap)
        exact = self.evaluator.eval_exact_match(p_sql, g_sql)
        return ExampleScore(hardness, int(exec_score), int(exact))


def _empty_sql() -> dict:
    import copy

    return copy.deepcopy(_EMPTY_SQL)
