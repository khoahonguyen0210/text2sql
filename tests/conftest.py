import json
import sqlite3
from pathlib import Path

import pytest

from text2sql.paths import EVALUATOR_DIR

TOY_TABLES = {
    "db_id": "toy",
    "table_names_original": ["singer", "concert"],
    "table_names": ["singer", "concert"],
    "column_names_original": [
        [-1, "*"], [0, "singer_id"], [0, "name"], [0, "country"], [0, "age"],
        [1, "concert_id"], [1, "singer_id"], [1, "year"],
    ],
    "column_names": [
        [-1, "*"], [0, "singer id"], [0, "name"], [0, "country"], [0, "age"],
        [1, "concert id"], [1, "singer id"], [1, "year"],
    ],
    "column_types": ["text", "number", "text", "text", "number", "number", "number", "number"],
    "primary_keys": [1, 5],
    "foreign_keys": [[6, 1]],
}


def build_toy_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER);
        CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, singer_id INTEGER, year INTEGER,
            FOREIGN KEY (singer_id) REFERENCES singer(singer_id));
        INSERT INTO singer VALUES (1, 'Joe Sharp', 'Netherlands', 52),
            (2, 'Timbaland', 'United States', 32), (3, 'Justin Brown', 'France', 29),
            (4, 'Rose White', 'France', 41);
        INSERT INTO concert VALUES (1, 1, 2014), (2, 2, 2014), (3, 2, 2015), (4, 3, 2015);
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def toy_root(tmp_path: Path) -> Path:
    """A Spider-style folder: tables.json and database/toy/toy.sqlite."""
    build_toy_db(tmp_path / "database" / "toy" / "toy.sqlite")
    (tmp_path / "tables.json").write_text(json.dumps([TOY_TABLES]), encoding="utf-8")
    return tmp_path


@pytest.fixture
def toy_db(toy_root: Path) -> Path:
    return toy_root / "database" / "toy" / "toy.sqlite"


needs_evaluator = pytest.mark.skipif(
    not (EVALUATOR_DIR / "evaluation.py").exists(),
    reason="run scripts/fetch_evaluator.py first",
)
