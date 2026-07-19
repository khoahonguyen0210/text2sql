import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from text2sql.models.prompt import PromptConfig


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    dtype: str = "float32"
    load_in_4bit: bool = False


@dataclass(frozen=True)
class DecodingConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.0


@dataclass(frozen=True)
class GenerationConfig:
    name: str
    model: ModelConfig
    prompt: PromptConfig = field(default_factory=PromptConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    batch_size: int = 1


def load_toml(path: Path) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def generation_config(path: Path) -> GenerationConfig:
    raw = load_toml(path)
    return GenerationConfig(
        name=raw["name"],
        model=ModelConfig(**raw["model"]),
        prompt=PromptConfig(**raw.get("prompt", {})),
        decoding=DecodingConfig(**raw.get("decoding", {})),
        batch_size=raw.get("batch_size", 1),
    )
