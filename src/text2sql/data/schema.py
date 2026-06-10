import re
from dataclasses import dataclass, field

SCHEMA_STYLES = ("ddl", "compact")
_SIMPLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    is_primary: bool = False


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Schema:
    db_id: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...] = field(default_factory=tuple)

    def table(self, name: str) -> Table | None:
        lowered = name.lower()
        return next((t for t in self.tables if t.name.lower() == lowered), None)


def quote(name: str) -> str:
    if _SIMPLE_NAME.match(name):
        return name
    return '"' + name.replace('"', '""') + '"'


def _primary_ids(raw: list) -> set[int]:
    # Newer tables.json files store composite keys as nested lists.
    ids: set[int] = set()
    for item in raw:
        if isinstance(item, list):
            ids.update(item)
        else:
            ids.add(item)
    return ids


def schema_from_spider(entry: dict) -> Schema:
    """Build a Schema from one tables.json entry, using the original names."""
    table_names = entry["table_names_original"]
    columns = entry["column_names_original"]
    types = entry["column_types"]
    primary = _primary_ids(entry.get("primary_keys", []))

    per_table: list[list[Column]] = [[] for _ in table_names]
    for idx, (table_idx, name) in enumerate(columns):
        if table_idx < 0:
            continue  # the "*" column
        per_table[table_idx].append(Column(name, types[idx], idx in primary))

    fks = []
    for col_idx, ref_idx in entry.get("foreign_keys", []):
        t_idx, col = columns[col_idx]
        r_idx, ref_col = columns[ref_idx]
        fks.append(ForeignKey(table_names[t_idx], col, table_names[r_idx], ref_col))

    tables = tuple(Table(name, tuple(cols)) for name, cols in zip(table_names, per_table, strict=True))
    return Schema(entry["db_id"], tables, tuple(fks))


def _ddl(schema: Schema, include_fks: bool) -> str:
    blocks = []
    for table in schema.tables:
        lines = [f"  {quote(c.name)} {c.type}" for c in table.columns]
        keys = [quote(c.name) for c in table.columns if c.is_primary]
        if keys:
            lines.append(f"  PRIMARY KEY ({', '.join(keys)})")
        if include_fks:
            for fk in schema.foreign_keys:
                if fk.table == table.name:
                    lines.append(
                        f"  FOREIGN KEY ({quote(fk.column)}) REFERENCES "
                        f"{quote(fk.ref_table)}({quote(fk.ref_column)})"
                    )
        blocks.append(f"CREATE TABLE {quote(table.name)} (\n" + ",\n".join(lines) + "\n);")
    return "\n\n".join(blocks)


def _compact(schema: Schema, include_fks: bool) -> str:
    lines = []
    for table in schema.tables:
        cols = ", ".join(
            f"{quote(c.name)}{' (key)' if c.is_primary else ''}" for c in table.columns
        )
        lines.append(f"{quote(table.name)}: {cols}")
    if include_fks and schema.foreign_keys:
        lines.append("Links:")
        lines.extend(
            f"{quote(fk.table)}.{quote(fk.column)} = {quote(fk.ref_table)}.{quote(fk.ref_column)}"
            for fk in schema.foreign_keys
        )
    return "\n".join(lines)


def serialize_schema(schema: Schema, style: str = "ddl", include_fks: bool = True) -> str:
    """Turn a schema into prompt text. The output depends only on the schema."""
    if style == "ddl":
        return _ddl(schema, include_fks)
    if style == "compact":
        return _compact(schema, include_fks)
    raise ValueError(f"unknown schema style: {style}")
