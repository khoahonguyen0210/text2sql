"""Kaggle entry point: install pinned packages, then run one plan from bundle.json."""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

PLAN = "smoke"  # set per notebook by scripts/kaggle_bundle.py
PACKAGES = [
    "transformers==5.20.0",
    "peft==0.21.2",
    "accelerate==1.15.0",
    "bitsandbytes==0.50.2",
    "sqlglot==30.23.0",
]
WORK = Path("/kaggle/working")


def find_bundle() -> Path:
    for path in Path("/kaggle/input").rglob("bundle.json"):
        return path.parent
    sys.exit("bundle.json not found; attach the bundle dataset")


def check_runtime() -> None:
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    print(gpu.stdout or "no GPU found", flush=True)
    try:
        socket.create_connection(("pypi.org", 443), timeout=10).close()
    except OSError:
        sys.exit("No internet. Kaggle turns it off until the account phone number is verified.")
    if gpu.returncode != 0:
        sys.exit("No GPU attached to this session.")


def main() -> None:
    check_runtime()
    bundle = find_bundle()
    spec = json.loads((bundle / "bundle.json").read_text())
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
    # The image's old torchao makes PEFT refuse to load adapters; it is not used here.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

    repo = WORK / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(bundle, repo)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(repo / "src"),
        "TEXT2SQL_ARTIFACTS_DIR": str(WORK / "artifacts"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    for step in spec["plans"][PLAN]:
        print(f"== {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, *step], cwd=repo, env=env, check=True)

    for ckpt in (WORK / "artifacts").rglob("checkpoints"):
        shutil.rmtree(ckpt)  # adapters are kept; optimizer states are not needed
    shutil.rmtree(repo)
    (WORK / "bundle.json").write_text(json.dumps({"plan": PLAN, "commit": spec["commit"]}))


if __name__ == "__main__":
    main()
