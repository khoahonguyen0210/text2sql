"""Start the API on 127.0.0.1. Usage: python -m text2sql.serving [--base] [--port 8000]"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from text2sql.config import generation_config, load_toml
from text2sql.data.schema import Schema
from text2sql.data.spider import database_path, load_questions, load_schemas
from text2sql.paths import CONFIGS_DIR, PROCESSED_DIR, ROOT
from text2sql.serving.app import Backend, Generated, Limits, create_app
from text2sql.serving.schemas import ModelInfo


class ModelBackend(Backend):
    def __init__(self, config_path: Path, adapter: Path | None):
        from text2sql.models.generate import Decoding, Generator
        from text2sql.models.loading import ModelSpec

        self.cfg = generation_config(config_path)
        spec = ModelSpec(self.cfg.model.id, self.cfg.model.revision, self.cfg.model.dtype,
                         str(adapter) if adapter else None)
        self.generator = Generator(spec, Decoding(**asdict(self.cfg.decoding)))
        self.info = ModelInfo(id=spec.model_id, revision=spec.revision,
                              adapter=adapter.name if adapter else None,
                              prompt_version=self.cfg.prompt.version)

    def generate(self, question: str, schema: Schema) -> Generated:
        from text2sql.models.prompt import build_messages

        out = self.generator.generate([build_messages(question, schema, self.cfg.prompt)])[0]
        return Generated(out.text, out.prompt_tokens, out.completion_tokens, out.latency_ms)


def allowed_databases(raw: dict, db_root: Path) -> dict[str, Path]:
    if raw.get("databases") == "dev":
        names = {e.db_id for e in load_questions(PROCESSED_DIR / "dev_questions.jsonl")}
    else:
        names = set(raw["databases"])
    return {name: database_path(name, db_root) for name in sorted(names)}


def build(config_path: Path, use_adapter: bool):
    raw = load_toml(config_path)
    adapter = ROOT / raw["adapter"] if use_adapter else None
    if adapter and not (adapter / "adapter_config.json").exists():
        sys.exit(f"adapter not found at {raw['adapter']}; run with --base or download it")
    db_paths = allowed_databases(raw, ROOT / raw["db_root"])
    schemas = load_schemas()
    return create_app(lambda: ModelBackend(config_path, adapter), schemas, db_paths,
                      Limits(**raw.get("limits", {})))


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "serve.toml")
    parser.add_argument("--base", action="store_true", help="serve the base model, no adapter")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(build(args.config, not args.base), host="127.0.0.1", port=args.port,
                log_level="info", server_header=False)


if __name__ == "__main__":
    main()
