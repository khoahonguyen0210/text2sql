import time
from dataclasses import dataclass

import torch

from text2sql.models.loading import ModelSpec, load_model, load_tokenizer


@dataclass(frozen=True)
class Decoding:
    max_new_tokens: int = 256
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.0


@dataclass
class Output:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class Generator:
    """Holds one loaded model and turns chat messages into text with fixed decoding."""

    def __init__(self, spec: ModelSpec, decoding: Decoding):
        self.spec = spec
        self.decoding = decoding
        self.tokenizer = load_tokenizer(spec)
        self.model = load_model(spec)
        self.device = next(self.model.parameters()).device

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def generate(self, batch: list[list[dict]]) -> list[Output]:
        texts = [self.render(m) for m in batch]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        start = time.perf_counter()
        out = self.model.generate(
            **enc,
            max_new_tokens=self.decoding.max_new_tokens,
            do_sample=self.decoding.do_sample,
            num_beams=self.decoding.num_beams,
            repetition_penalty=self.decoding.repetition_penalty,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        prompt_len = enc["input_ids"].shape[1]
        results = []
        for i in range(len(batch)):
            new_ids = out[i, prompt_len:]
            completion = new_ids[new_ids != self.tokenizer.pad_token_id]
            results.append(
                Output(
                    text=self.tokenizer.decode(completion, skip_special_tokens=True),
                    prompt_tokens=int(enc["attention_mask"][i].sum()),
                    completion_tokens=int(completion.numel()),
                    latency_ms=round(elapsed_ms / len(batch), 1),
                )
            )
        return results
