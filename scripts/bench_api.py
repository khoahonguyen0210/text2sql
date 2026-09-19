"""Measure API latency on this computer.

Starts the API, waits until the model is loaded, sends warm-up requests, then timed requests
with fixed dev questions. Results go to results/latency/.
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from text2sql.data.spider import load_questions
from text2sql.manifest import hardware_info, package_versions, utc_now, write_json
from text2sql.paths import PROCESSED_DIR, RESULTS_DIR, ROOT


def post(url: str, body: dict, timeout: float = 600) -> tuple[int, dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
            status = resp.status
    except urllib.error.HTTPError as e:
        status, payload = e.code, {}
    return status, payload, (time.perf_counter() - start) * 1000


def wait_ready(base: str, limit_s: float) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < limit_s:
        try:
            with urllib.request.urlopen(base + "/ready", timeout=5) as resp:
                if resp.status == 200:
                    return time.perf_counter() - start
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise SystemExit("API did not become ready in time")


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))], 1)


def run_load(base: str, questions, concurrency: int) -> dict:
    def one(ex):
        return post(base + "/v1/generate-sql", {"question": ex.question, "db_id": ex.db_id})

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, questions))
    wall = time.perf_counter() - start
    ok = [(s, p, ms) for s, p, ms in results if s == 200]
    client_ms = [ms for _, _, ms in ok]
    model_ms = [p["latency_ms"] for _, p, _ in ok]
    out_tokens = [p["completion_tokens"] for _, p, _ in ok]
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "client_latency_ms": {"p50": pct(client_ms, 0.5), "p95": pct(client_ms, 0.95),
                              "mean": round(statistics.mean(client_ms), 1)},
        "model_latency_ms": {"p50": pct(model_ms, 0.5), "p95": pct(model_ms, 0.95)},
        "throughput_rps": round(len(ok) / wall, 3),
        "completion_tokens_mean": round(statistics.mean(out_tokens), 1),
        "prompt_tokens_mean": round(statistics.mean(p["prompt_tokens"] for _, p, _ in ok), 1),
        "wall_seconds": round(wall, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--base", action="store_true", help="serve the base model, no adapter")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "latency")
    args = parser.parse_args()

    questions = load_questions(PROCESSED_DIR / "dev_questions.jsonl")
    sample = random.Random(args.seed).sample(questions, args.requests + args.warmup)
    warmup, timed = sample[: args.warmup], sample[args.warmup :]
    base = f"http://127.0.0.1:{args.port}"
    cmd = [sys.executable, "-m", "text2sql.serving", "--port", str(args.port)]
    if args.base:
        cmd.append("--base")
    server = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ready_s = wait_ready(base, 900)
        for ex in warmup:
            post(base + "/v1/generate-sql", {"question": ex.question, "db_id": ex.db_id})
        runs = [run_load(base, timed, c) for c in args.concurrency]
    finally:
        server.terminate()
        server.wait(timeout=60)

    name = "base" if args.base else "tuned"
    report = {
        "measured_at": utc_now(),
        "model": name,
        "setup": {
            "where": "one laptop CPU, API and client on the same machine",
            "dtype": "float32, adapter merged into the base weights" if not args.base
            else "float32",
            "requests_per_level": args.requests,
            "warmup_requests": args.warmup,
            "questions": f"random sample of Spider dev questions, seed {args.seed}",
            "arrival": "closed loop: each worker sends its next request after the last reply",
            "timed_part": "client time covers HTTP, prompt building, generation, SQL "
                          "extraction and the safety check; it does not run the SQL",
            "max_new_tokens": 256,
        },
        "startup_seconds_until_ready": round(ready_s, 1),
        "runs": runs,
        "hardware": hardware_info(),
        "packages": package_versions(),
    }
    write_json(args.out / f"{name}.json", report)
    print(json.dumps({k: report[k] for k in ("startup_seconds_until_ready", "runs")}, indent=2))


if __name__ == "__main__":
    main()
