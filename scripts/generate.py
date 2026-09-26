"""Generate SQL for a Spider split and save one prediction per example.

The run can be stopped and started again: finished IDs are skipped.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from text2sql.config import generation_config
from text2sql.data.spider import get_schema, load_questions, load_schemas
from text2sql.evaluation.extract import extract_sql
from text2sql.evaluation.records import Prediction, append_prediction, read_predictions
from text2sql.manifest import environment, new_run_id, read_json, sha256_file, utc_now, write_json
from text2sql.models.generate import Decoding, Generator
from text2sql.models.loading import ModelSpec
from text2sql.models.prompt import build_messages
from text2sql.paths import ARTIFACTS_DIR, PROCESSED_DIR, SPIDER_DIR, relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=PROCESSED_DIR / "dev_questions.jsonl")
    parser.add_argument("--limit", type=int, help="first N examples only (a smoke subset)")
    parser.add_argument("--adapter", type=Path, help="folder with a trained LoRA adapter")
    parser.add_argument("--dtype", help="override the dtype in the config")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--out-root", type=Path, default=ARTIFACTS_DIR / "runs")
    args = parser.parse_args()

    cfg = generation_config(args.config)
    run_id = args.run_id or new_run_id(cfg.name)
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"
    manifest_file = out_dir / "manifest.json"

    examples = load_questions(args.questions)
    if args.limit:
        examples = examples[: args.limit]
    schemas = load_schemas()
    done = {p.id for p in read_predictions(pred_file)} if pred_file.exists() else set()
    todo = [e for e in examples if e.id not in done]

    spec = ModelSpec(
        model_id=cfg.model.id,
        revision=cfg.model.revision,
        dtype=args.dtype or cfg.model.dtype,
        adapter=str(args.adapter) if args.adapter else None,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    decoding = Decoding(**asdict(cfg.decoding))
    batch_size = args.batch_size or cfg.batch_size
    manifest = read_json(manifest_file) if manifest_file.exists() else {
        "run_id": run_id,
        "started_at": utc_now(),
        "task": "generate",
        "split": examples[0].split,
        "subset": f"first {args.limit}" if args.limit else "full",
        "example_count": len(examples),
        "data": {
            "questions_file": relative(args.questions),
            "questions_sha256": sha256_file(args.questions),
            "tables_json_sha256": sha256_file(SPIDER_DIR / "tables.json"),
        },
        "model": {"id": spec.model_id, "revision": spec.revision, "dtype": spec.dtype,
                  "adapter": args.adapter.parent.name if args.adapter else None,
                  "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors")
                  if args.adapter else None},
        "prompt": asdict(cfg.prompt) | {"hash": cfg.prompt.hash()},
        "decoding": asdict(decoding),
        "batch_size": batch_size,
        "config_file": relative(args.config),
        "environment": environment(),
    }
    write_json(manifest_file, manifest)

    print(f"{run_id}: {len(todo)} to generate, {len(done)} already done")
    load_start = time.perf_counter()
    generator = Generator(spec, decoding)
    manifest["model_load_seconds"] = round(time.perf_counter() - load_start, 1)
    gen_start = time.perf_counter()
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        messages = [build_messages(e.question, get_schema(schemas, e.db_id), cfg.prompt)
                    for e in batch]
        for ex, out in zip(batch, generator.generate(messages), strict=True):
            sql, status = extract_sql(out.text)
            append_prediction(pred_file, Prediction(
                id=ex.id, db_id=ex.db_id, raw_output=out.text, sql=sql,
                extraction_status=status, run_id=run_id, model_id=spec.model_id,
                prompt_version=cfg.prompt.version, latency_ms=out.latency_ms,
                prompt_tokens=out.prompt_tokens, completion_tokens=out.completion_tokens,
            ))
        print(f"  {min(i + batch_size, len(todo))}/{len(todo)}", flush=True)

    preds = read_predictions(pred_file)
    manifest.update({
        "finished_at": utc_now(),
        "generation_seconds": round(time.perf_counter() - gen_start, 1),
        "prediction_count": len(preds),
        "predictions_sha256": sha256_file(pred_file),
    })
    write_json(manifest_file, manifest)
    print(json.dumps({k: manifest[k] for k in ("run_id", "prediction_count")}))


if __name__ == "__main__":
    main()
