import json
from dataclasses import dataclass
from pathlib import Path

from text2sql.data.schema import Schema, schema_from_spider
from text2sql.paths import SPIDER_DIR

SPLIT_FILES = {
    "train": ("train_spider.json", "train_others.json"),
    "dev": ("dev.json",),
}


@dataclass(frozen=True)
class Example:
    id: str
    split: str
    db_id: str
    question: str
    query: str


class UnknownDatabaseError(KeyError):
    pass


def load_examples(split: str, spider_dir: Path = SPIDER_DIR) -> list[Example]:
    """Load one official split. IDs follow the order of the official files."""
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split: {split}")
    examples: list[Example] = []
    for name in SPLIT_FILES[split]:
        path = spider_dir / name
        if not path.exists():
            if name == "train_others.json":
                continue
            raise FileNotFoundError(f"missing {path.name}; run scripts/prepare_spider.py")
        for row in json.loads(path.read_text(encoding="utf-8")):
            examples.append(
                Example(
                    id=f"{split}-{len(examples):05d}",
                    split=split,
                    db_id=row["db_id"],
                    question=" ".join(row["question"].split()),
                    query=row["query"].strip(),
                )
            )
    return examples


def load_questions(path: Path, split: str = "dev") -> list[Example]:
    """Questions without gold SQL, as written by scripts/build_data.py."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [Example(r["id"], split, r["db_id"], r["question"], "") for r in rows]


def load_schemas(spider_dir: Path = SPIDER_DIR) -> dict[str, Schema]:
    entries = json.loads((spider_dir / "tables.json").read_text(encoding="utf-8"))
    return {e["db_id"]: schema_from_spider(e) for e in entries}


def get_schema(schemas: dict[str, Schema], db_id: str) -> Schema:
    try:
        return schemas[db_id]
    except KeyError:
        raise UnknownDatabaseError(db_id) from None


def database_path(db_id: str, db_root: Path) -> Path:
    """Path of the main SQLite file for db_id. The name must be a known folder."""
    if not db_id or "/" in db_id or "\\" in db_id or db_id.startswith("."):
        raise UnknownDatabaseError(db_id)
    path = db_root / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise UnknownDatabaseError(db_id)
    return path
