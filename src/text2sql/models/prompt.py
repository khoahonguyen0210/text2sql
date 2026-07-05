import json
from dataclasses import dataclass

from text2sql.data.schema import Schema, serialize_schema
from text2sql.manifest import sha256_text

TEMPLATES = {
    "v1": {
        "system": (
            "You are an expert in SQLite. Write one SQLite query that answers the question. "
            "Use only the tables and columns in the schema. Reply with the SQL query only."
        ),
        "user": "Database schema:\n\n{schema}\n\nQuestion: {question}",
    },
}


@dataclass(frozen=True)
class PromptConfig:
    version: str = "v1"
    schema_style: str = "ddl"
    include_foreign_keys: bool = True

    def hash(self) -> str:
        payload = {"template": TEMPLATES[self.version], **self.__dict__}
        return sha256_text(json.dumps(payload, sort_keys=True))[:16]


def build_messages(question: str, schema: Schema, config: PromptConfig) -> list[dict]:
    """Chat messages for one question. The gold SQL is never part of the prompt."""
    template = TEMPLATES[config.version]
    schema_text = serialize_schema(schema, config.schema_style, config.include_foreign_keys)
    return [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": template["user"].format(schema=schema_text, question=question)},
    ]
