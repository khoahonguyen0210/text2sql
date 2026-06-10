import json
from pathlib import Path

import pytest

from tests.conftest import TOY_TABLES
from text2sql.data.schema import quote, schema_from_spider, serialize_schema
from text2sql.data.spider import (
    UnknownDatabaseError,
    database_path,
    get_schema,
    load_examples,
    load_schemas,
)


def test_ddl_has_keys_and_links():
    text = serialize_schema(schema_from_spider(TOY_TABLES))
    assert "CREATE TABLE singer (" in text
    assert "PRIMARY KEY (singer_id)" in text
    assert "FOREIGN KEY (singer_id) REFERENCES singer(singer_id)" in text


def test_links_can_be_left_out():
    schema = schema_from_spider(TOY_TABLES)
    assert "FOREIGN KEY" not in serialize_schema(schema, include_fks=False)
    assert "Links:" not in serialize_schema(schema, "compact", include_fks=False)
    assert "concert.singer_id = singer.singer_id" in serialize_schema(schema, "compact")


def test_output_is_deterministic():
    a = serialize_schema(schema_from_spider(TOY_TABLES))
    b = serialize_schema(schema_from_spider(json.loads(json.dumps(TOY_TABLES))))
    assert a == b


def test_unusual_names_are_quoted():
    assert quote("Home Town") == '"Home Town"'
    assert quote("%_Change_2007") == '"%_Change_2007"'
    assert quote('a"b') == '"a""b"'
    assert quote("singer_id") == "singer_id"


def test_composite_primary_key():
    entry = dict(TOY_TABLES, primary_keys=[[5, 6]])
    text = serialize_schema(schema_from_spider(entry))
    assert "PRIMARY KEY (concert_id, singer_id)" in text


def test_unknown_style_fails():
    with pytest.raises(ValueError):
        serialize_schema(schema_from_spider(TOY_TABLES), "yaml")


def test_unknown_database(toy_root: Path):
    schemas = load_schemas(toy_root)
    assert get_schema(schemas, "toy").db_id == "toy"
    with pytest.raises(UnknownDatabaseError):
        get_schema(schemas, "missing")


@pytest.mark.parametrize("db_id", ["", "../toy", "..", "toy/../toy", "C:\\x", "missing"])
def test_database_path_rejects_bad_ids(toy_root: Path, db_id: str):
    with pytest.raises(UnknownDatabaseError):
        database_path(db_id, toy_root / "database")


def test_examples_get_stable_ids(tmp_path: Path):
    row = {"db_id": "toy", "question": "How  many\nsingers?", "query": "SELECT 1"}
    (tmp_path / "dev.json").write_text(json.dumps([row] * 3), encoding="utf-8")
    examples = load_examples("dev", tmp_path)
    assert [e.id for e in examples] == ["dev-00000", "dev-00001", "dev-00002"]
    assert examples[0].question == "How many singers?"
