import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from text2sql.paths import ROOT

_PACKAGES = (
    "torch", "transformers", "peft", "accelerate", "bitsandbytes",
    "tokenizers", "sqlglot", "nltk", "sqlparse", "fastapi",
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(name: str) -> str:
    return f"{name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_state() -> dict:
    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return out.stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_info() -> dict:
    """CPU and GPU facts only. No host name, user name or paths."""
    info: dict = {
        "os": platform.platform(terse=True),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": platform.processor() or None,
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["cuda"] = torch.version.cuda
            info["gpu"] = props.name
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
    except ImportError:
        pass
    return info


def environment() -> dict:
    return {"hardware": hardware_info(), "packages": package_versions(), "git": git_state()}


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
