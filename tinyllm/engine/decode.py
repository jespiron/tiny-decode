import time

import torch

from .model import LoadedModel


@torch.inference_mode()
def generate(
    lm: LoadedModel, prompt: str, max_new_tokens: int = 64
) -> tuple[str, list[float]]:
    tok = lm.tokenizer
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(lm.device)
    eos_id = tok.eos_token_id
    per_token_seconds: list[float] = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = lm.model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    all_ids = torch.cat([input_ids, next_id], dim=1)
    torch.cuda.synchronize()
    per_token_seconds.append(time.perf_counter() - t0)

    if eos_id is not None and next_id.item() == eos_id:
        return tok.decode(all_ids[0], skip_special_tokens=True), per_token_seconds

    for _ in range(max_new_tokens - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = lm.model(input_ids=next_id, past_key_values=cache, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_ids = torch.cat([all_ids, next_id], dim=1)
        torch.cuda.synchronize()
        per_token_seconds.append(time.perf_counter() - t0)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return tok.decode(all_ids[0], skip_special_tokens=True), per_token_seconds
