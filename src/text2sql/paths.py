import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


DATA_DIR = _env_path("TEXT2SQL_DATA_DIR", ROOT / "data")
ARTIFACTS_DIR = _env_path("TEXT2SQL_ARTIFACTS_DIR", ROOT / "artifacts")
EXTERNAL_DIR = _env_path("TEXT2SQL_EXTERNAL_DIR", ROOT / "external")

RAW_DIR = DATA_DIR / "raw"
SPIDER_DIR = DATA_DIR / "spider"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATOR_DIR = EXTERNAL_DIR / "test-suite-sql-eval"
NLTK_DIR = EXTERNAL_DIR / "nltk_data"
RESULTS_DIR = ROOT / "results"
CONFIGS_DIR = ROOT / "configs"


def relative(path: Path) -> str:
    """Path relative to the repo root, so manifests never hold local folder names."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name
