"""Build the training set and the dev question file, with dedupe, a leakage audit and a manifest."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from text2sql.data.build import (
    TRANSFORM_VERSION,
    dedupe,
    leakage_audit,
    length_stats,
    per_db_counts,
)
from text2sql.data.spider import get_schema, load_examples, load_schemas
from text2sql.manifest import git_state, read_json, sha256_file, utc_now, write_json
from text2sql.models.prompt import PromptConfig, build_messages
from text2sql.paths import PROCESSED_DIR, SPIDER_DIR

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_lengths(examples, schemas, prompt: PromptConfig, with_target: bool) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    lengths = []
    for ex in examples:
        messages = build_messages(ex.question, get_schema(schemas, ex.db_id), prompt)
        if with_target:
            messages.append({"role": "assistant", "content": ex.query})
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=not with_target)
        lengths.append(len(tok(text)["input_ids"]))
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-lengths", action="store_true", help="skip the tokenizer pass")
    args = parser.parse_args()

    schemas = load_schemas()
    train_raw, dev = load_examples("train"), load_examples("dev")
    for ex in train_raw + dev:
        get_schema(schemas, ex.db_id)  # every example must resolve to its own schema

    train, report = dedupe(train_raw)
    audit = leakage_audit(train, dev)
    removed = set(audit["removed_ids"])
    train = [e for e in train if e.id not in removed]

    write_jsonl(PROCESSED_DIR / "train.jsonl",
                [{"id": e.id, "db_id": e.db_id, "question": e.question, "query": e.query}
                 for e in train])
    write_jsonl(PROCESSED_DIR / "dev_questions.jsonl",
                [{"id": e.id, "db_id": e.db_id, "question": e.question} for e in dev])

    prompt = PromptConfig()
    lengths = {}
    if not args.skip_lengths:
        lengths = {
            "prompt": asdict(prompt) | {"hash": prompt.hash()},
            "tokenizer": f"{MODEL_ID}@{MODEL_REVISION[:12]}",
            "train_prompt_plus_target": length_stats(token_lengths(train, schemas, prompt, True)),
            "dev_prompt": length_stats(token_lengths(dev, schemas, prompt, False)),
        }

    source = SPIDER_DIR / "SOURCE.json"
    manifest = {
        "created_at": utc_now(),
        "transform_version": TRANSFORM_VERSION,
        "code": git_state(),
        "source": read_json(source) if source.exists() else "unknown",
        "inputs_sha256": {name: sha256_file(SPIDER_DIR / name)
                          for name in ("train_spider.json", "train_others.json", "dev.json",
                                       "tables.json") if (SPIDER_DIR / name).exists()},
        "steps": {
            "train_loaded": len(train_raw),
            "dedupe": asdict(report),
            "removed_cross_split_pairs": len(removed),
            "train_final": len(train),
            "dev_questions": len(dev),
        },
        "split_ids": {
            "train_first": train[0].id, "train_last": train[-1].id,
            "dev_first": dev[0].id, "dev_last": dev[-1].id,
        },
        "databases": {"train": len(per_db_counts(train)), "dev": len(per_db_counts(dev))},
        "leakage_audit": audit,
        "token_lengths": lengths,
        "outputs_sha256": {
            "train.jsonl": sha256_file(PROCESSED_DIR / "train.jsonl"),
            "dev_questions.jsonl": sha256_file(PROCESSED_DIR / "dev_questions.jsonl"),
        },
        "notes": [
            "dev_questions.jsonl has no gold SQL, so generation never sees it.",
            "Dedupe removes only exact repeats of (db_id, question, SQL).",
            "Train rows whose question and SQL both equal a dev pair are removed.",
        ],
    }
    write_json(PROCESSED_DIR / "manifest.json", manifest)
    print(json.dumps(manifest["steps"], indent=2))
    print(json.dumps({k: v for k, v in audit.items() if not k.startswith("examples")}, indent=2))
    if lengths:
        print(json.dumps(lengths, indent=2))


if __name__ == "__main__":
    main()
