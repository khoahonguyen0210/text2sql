from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    dtype: str = "float32"
    adapter: str | None = None  # folder with a saved LoRA adapter
    load_in_4bit: bool = False


def quantization_config(compute_dtype: str = "float16"):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=DTYPES[compute_dtype],
    )


def load_tokenizer(spec: ModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(spec: ModelSpec):
    """Load the base model once, and the LoRA adapter on top if one is given."""
    kwargs: dict = {"revision": spec.revision, "dtype": DTYPES[spec.dtype]}
    if spec.load_in_4bit:
        kwargs["quantization_config"] = quantization_config(spec.dtype)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    if spec.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, Path(spec.adapter))
        if not spec.load_in_4bit:
            model = model.merge_and_unload()
    if torch.cuda.is_available() and not spec.load_in_4bit:
        model = model.to("cuda")
    model.eval()
    return model
