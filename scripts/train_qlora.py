"""Train a LoRA adapter on the Spider train split, with the base model in 4-bit (QLoRA).

--smoke runs a few steps without 4-bit, so the whole path can be checked on a CPU.
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from text2sql.config import load_toml
from text2sql.data.spider import load_schemas
from text2sql.evaluation.extract import one_line
from text2sql.manifest import environment, new_run_id, sha256_file, utc_now, write_json
from text2sql.models.prompt import PromptConfig, build_messages
from text2sql.models.train import (
    Collator,
    LoraSettings,
    TrainSettings,
    encode_all,
    lora_b_norm,
)
from text2sql.paths import ARTIFACTS_DIR, PROCESSED_DIR, relative


def load_records(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["query"] = one_line(row["query"])
    return rows[:limit] if limit else rows


def build_model(model_cfg: dict, quant: dict, lora: LoraSettings, smoke: bool):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if smoke:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["id"], revision=model_cfg["revision"], dtype=torch.float32
        )
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant["quant_type"],
            bnb_4bit_use_double_quant=quant["double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["id"], revision=model_cfg["revision"], quantization_config=bnb,
            dtype=getattr(torch, quant["compute_dtype"]), device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM", r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules), bias="none",
    )
    return get_peft_model(model, peft_cfg)


def check_reload(model_cfg: dict, adapter_dir: Path, record: dict, schemas, prompt) -> str:
    """Load the saved adapter into a fresh base model and generate once."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_cfg["revision"])
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"], dtype=dtype
    )
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    if torch.cuda.is_available():
        model = model.to("cuda")
    messages = build_messages(record["question"], schemas[record["db_id"]], prompt)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=64, do_sample=False, temperature=None,
                         top_p=None, top_k=None)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, default=PROCESSED_DIR / "train.jsonl")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, help="first N training rows only")
    parser.add_argument("--resume", type=Path, help="checkpoint folder to resume from")
    parser.add_argument("--run-id")
    parser.add_argument("--out-root", type=Path, default=ARTIFACTS_DIR / "runs")
    args = parser.parse_args()

    raw = load_toml(args.config)
    model_cfg, quant = raw["model"], raw["quantization"]
    lora = LoraSettings(**{**raw["lora"], "target_modules": tuple(raw["lora"]["target_modules"])})
    settings = TrainSettings(**raw["train"], max_steps=args.max_steps)
    if args.smoke:
        settings = TrainSettings(**{**asdict(settings), "batch_size": 1, "grad_accum": 1,
                                    "max_steps": max(args.max_steps, 3), "logging_steps": 1})
    prompt = PromptConfig(**raw["prompt"])
    run_id = args.run_id or new_run_id(raw["name"] + ("-smoke" if args.smoke else ""))
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(settings.seed)
    schemas = load_schemas()
    records = load_records(args.train_file, args.limit)
    tok = AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_cfg["revision"])
    encoded = encode_all(tok, records, schemas, prompt, settings.max_seq_len)
    lengths = [len(r["input_ids"]) for r in encoded.rows]

    manifest = {
        "run_id": run_id,
        "task": "train",
        "smoke": args.smoke,
        "started_at": utc_now(),
        "config_file": relative(args.config),
        "model": model_cfg,
        "quantization": None if args.smoke else quant,
        "lora": asdict(lora),
        "train": asdict(settings),
        "prompt": asdict(prompt) | {"hash": prompt.hash()},
        "data": {
            "train_file": relative(args.train_file),
            "train_sha256": sha256_file(args.train_file),
            "rows_in": len(records),
            "rows_used": len(encoded.rows),
            "dropped_too_long": encoded.dropped_too_long,
            "tokens_total": sum(lengths),
            "max_len_seen": max(lengths),
        },
        "checkpoint_rule": "final checkpoint; dev is not used to choose one",
        "environment": environment(),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest["data"] | {"dropped_too_long": len(encoded.dropped_too_long)}))

    model = build_model(model_cfg, quant, lora, args.smoke)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    b_norm_before = lora_b_norm(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=settings.epochs,
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        learning_rate=settings.learning_rate,
        lr_scheduler_type=settings.lr_scheduler,
        warmup_steps=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        save_total_limit=2,
        seed=settings.seed,
        data_seed=settings.seed,
        fp16=not args.smoke and quant["compute_dtype"] == "float16",
        gradient_checkpointing=not args.smoke,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length",
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        use_cpu=args.smoke and not torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=encoded.rows,
                      data_collator=Collator(tok.pad_token_id))
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    train_seconds = time.perf_counter() - start

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    history = [{k: h[k] for k in ("step", "loss", "learning_rate", "grad_norm") if k in h}
               for h in trainer.state.log_history if "loss" in h]
    losses = [h["loss"] for h in history]
    b_norm_after = lora_b_norm(model)
    del trainer, model
    if torch.cuda.is_available():
        peak_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        torch.cuda.empty_cache()
    else:
        peak_gb = None

    sample = check_reload(model_cfg, adapter_dir, records[0], schemas, prompt)
    manifest.update({
        "finished_at": utc_now(),
        "train_seconds": round(train_seconds, 1),
        "global_steps": result.global_step,
        "final_train_loss": result.training_loss,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_all_finite": all(math.isfinite(x) for x in losses),
        "loss_history": history,
        "trainable_params": trainable,
        "total_params": total,
        "lora_b_norm_before": b_norm_before,
        "lora_b_norm_after": b_norm_after,
        "adapter_updated": b_norm_after > b_norm_before,
        "peak_gpu_memory_gb": peak_gb,
        "adapter_dir": relative(adapter_dir),
        "adapter_files_sha256": {p.name: sha256_file(p) for p in sorted(adapter_dir.iterdir())},
        "reload_check": {"question": records[0]["question"], "output": sample},
    })
    write_json(out_dir / "manifest.json", manifest)
    keys = ("final_train_loss", "loss_all_finite", "adapter_updated", "train_seconds")
    print(json.dumps({k: manifest[k] for k in keys} | {"reload_output": sample}))


if __name__ == "__main__":
    main()
