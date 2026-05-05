import time
from contextlib import contextmanager

import torch

from .model import LoadedModel

@contextmanager
def _timed(per_step_seconds: list[float]):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    per_step_seconds.append(time.perf_counter() - t0)


def _extend(all_ids, attn_mask, next_ids, batch_size, device):
    all_ids = torch.cat([all_ids, next_ids], dim=1)
    attn_mask = torch.cat([attn_mask, torch.ones(batch_size, 1, device=device)], dim=1)
    return all_ids, attn_mask


def _record_eos(next_ids, eos_ids, done, eos_decode_step, step):
    newly_done = torch.isin(next_ids.squeeze(-1), eos_ids) & ~done
    for i in newly_done.nonzero(as_tuple=False).squeeze(-1).tolist():
        eos_decode_step[i] = step
    done |= newly_done


@torch.inference_mode()
def generate(
    lm: LoadedModel, prompt: str, max_new_tokens: int = 64
) -> tuple[str, list[float]]:
    texts, per_step, _ = generate_batch(lm, [prompt], max_new_tokens)
    return texts[0], per_step


@torch.inference_mode()
def generate_batch(
    lm: LoadedModel,
    prompts: list[str],
    max_new_tokens: int = 64,
    eos_token_ids: list[int] | None = None,
) -> tuple[list[str], list[float], list[int]]:
    tok = lm.tokenizer
    batch_size = len(prompts)

    if eos_token_ids is not None:
        eos_ids = torch.tensor(eos_token_ids, device=lm.device)
    elif tok.eos_token_id is not None:
        eos_ids = torch.tensor([tok.eos_token_id], device=lm.device)
    else:
        eos_ids = None

    # left-pad each prompt
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True)
    input_ids = enc.input_ids.to(lm.device)       # [B, L_pad]
    attn_mask = enc.attention_mask.to(lm.device)   # [B, L_pad]

    per_step_seconds: list[float] = []
    done = torch.zeros(batch_size, dtype=torch.bool, device=lm.device)
    # eos_decode_step[i]: which step seq i finished on (0 = finished during prefill).
    eos_decode_step: list[int] = [max_new_tokens] * batch_size

    # prefill stage
    with _timed(per_step_seconds):
        out = lm.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
    cache = out.past_key_values
    next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
    all_ids, attn_mask = _extend(input_ids, attn_mask, next_ids, batch_size, lm.device)
    if eos_ids is not None:
        _record_eos(next_ids, eos_ids, done, eos_decode_step, step=0)
    if done.all():
        return [tok.decode(row, skip_special_tokens=True) for row in all_ids], per_step_seconds, eos_decode_step

    # decode stage
    for step in range(max_new_tokens - 1):
        with _timed(per_step_seconds):
            out = lm.model(
                input_ids=next_ids,
                attention_mask=attn_mask,
                past_key_values=cache,
                use_cache=True,
            )
        cache = out.past_key_values
        next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_ids, attn_mask = _extend(all_ids, attn_mask, next_ids, batch_size, lm.device)
        if eos_ids is not None:
            _record_eos(next_ids, eos_ids, done, eos_decode_step, step=step + 1)
        if done.all():
            break

    total_decode_steps = len(per_step_seconds) - 1
    eos_decode_step = [min(s, total_decode_steps - 1) for s in eos_decode_step]

    return [tok.decode(row, skip_special_tokens=True) for row in all_ids], per_step_seconds, eos_decode_step
