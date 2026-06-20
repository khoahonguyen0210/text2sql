import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from text2sql.data.spider import Example


@dataclass
class Prediction:
    id: str
    db_id: str
    raw_output: str
    sql: str
    extraction_status: str
    run_id: str
    model_id: str
    prompt_version: str
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str = ""
    extra: dict = field(default_factory=dict)


def append_prediction(path: Path, pred: Prediction) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(pred), ensure_ascii=False) + "\n")
        f.flush()


def read_predictions(path: Path) -> list[Prediction]:
    preds = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(Prediction(**json.loads(line)))
    return preds


@dataclass
class ContractReport:
    expected: int
    received: int
    missing: list[str]
    duplicates: list[str]
    unknown: list[str]
    db_mismatch: list[str]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.duplicates or self.unknown or self.db_mismatch)


def check_contract(examples: list[Example], preds: list[Prediction]) -> ContractReport:
    """Check that there is exactly one prediction per example, with the right db_id."""
    by_id = {e.id: e for e in examples}
    counts = Counter(p.id for p in preds)
    return ContractReport(
        expected=len(examples),
        received=len(preds),
        missing=[e.id for e in examples if e.id not in counts],
        duplicates=sorted(i for i, n in counts.items() if n > 1),
        unknown=sorted(i for i in counts if i not in by_id),
        db_mismatch=sorted(p.id for p in preds if p.id in by_id and by_id[p.id].db_id != p.db_id),
    )


def align(examples: list[Example], preds: list[Prediction]) -> list[Prediction | None]:
    """Predictions in example order. Missing ones are None; the last duplicate wins."""
    by_id = {p.id: p for p in preds}
    return [by_id.get(e.id) for e in examples]
