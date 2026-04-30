from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    name: str


def load(name: str = "gpt2") -> LoadedModel:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Run via Modal.")
    device = torch.device("cuda")
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model, tokenizer=tokenizer, device=device, dtype=dtype, name=name
    )
