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


def load_model(name: str = "gpt2") -> LoadedModel:
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


@torch.inference_mode()
def generate(lm: LoadedModel, prompt: str, max_new_tokens: int = 64) -> str:
    tok = lm.tokenizer
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(lm.device)
    eos_id = tok.eos_token_id

    for _ in range(max_new_tokens):
        out = lm.model(input_ids=input_ids, use_cache=False)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return tok.decode(input_ids[0], skip_special_tokens=True)
