from tests.conftest import TOY_TABLES
from text2sql.data.schema import schema_from_spider
from text2sql.models.prompt import PromptConfig
from text2sql.models.train import IGNORE, Collator, encode_all, encode_example


class CharTokenizer:
    """One token per character, enough to check masking without model files."""

    eos_token = "#"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


SCHEMAS = {"toy": schema_from_spider(TOY_TABLES)}


def test_only_the_sql_is_trained_on():
    row = encode_example(CharTokenizer(), "How many?", "SELECT 1", SCHEMAS["toy"], PromptConfig())
    target = [ord(c) for c in "SELECT 1#"]
    trained = [t for t in row["labels"] if t != IGNORE]
    assert trained == target
    assert row["input_ids"][-len(target):] == target
    assert len(row["input_ids"]) == len(row["labels"])


def test_too_long_rows_are_dropped_not_cut():
    records = [
        {"id": "train-00000", "db_id": "toy", "question": "q", "query": "SELECT 1"},
        {"id": "train-00001", "db_id": "toy", "question": "q" * 5000, "query": "SELECT 1"},
    ]
    out = encode_all(CharTokenizer(), records, SCHEMAS, PromptConfig(), max_len=2000)
    assert len(out.rows) == 1
    assert out.dropped_too_long == ["train-00001"]


def test_collator_pads_labels_with_ignore():
    batch = Collator(pad_id=0)([
        {"input_ids": [5, 6, 7], "labels": [IGNORE, 6, 7]},
        {"input_ids": [5], "labels": [5]},
    ])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE, 6, 7], [5, IGNORE, IGNORE]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
