from dataclasses import dataclass, field

import torch

from text2sql.data.schema import Schema
from text2sql.models.prompt import PromptConfig, build_messages

IGNORE = -100


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )


@dataclass(frozen=True)
class TrainSettings:
    epochs: float = 2
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 2048
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    max_steps: int = -1


@dataclass
class EncodedSet:
    rows: list[dict] = field(default_factory=list)
    dropped_too_long: list[str] = field(default_factory=list)


def encode_example(tokenizer, question: str, sql: str, schema: Schema,
                   prompt: PromptConfig) -> dict:
    """Prompt tokens are masked, so the loss only covers the SQL and the end token."""
    messages = build_messages(question, schema, prompt)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(sql + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE] * len(prompt_ids) + target_ids,
    }


def encode_all(tokenizer, records: list[dict], schemas: dict[str, Schema],
               prompt: PromptConfig, max_len: int) -> EncodedSet:
    out = EncodedSet()
    for rec in records:
        row = encode_example(tokenizer, rec["question"], rec["query"], schemas[rec["db_id"]],
                             prompt)
        if len(row["input_ids"]) > max_len:
            out.dropped_too_long.append(rec["id"])  # dropped, never cut
            continue
        out.rows.append(row)
    return out


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [IGNORE] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def lora_b_norm(model) -> float:
    """LoRA B matrices start at zero, so a non-zero norm shows the adapter was trained."""
    total = 0.0
    for name, param in model.named_parameters():
        if "lora_B" in name:
            total += float(param.detach().float().norm() ** 2)
    return total ** 0.5
