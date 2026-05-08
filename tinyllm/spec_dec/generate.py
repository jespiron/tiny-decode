import time

import torch
from transformers import DynamicCache

from engine.model import LoadedModel
from .draft import run_draft
from .sampler import greedy_accept, rejection_sample
from .verify import run_verify

def _crop_cache(cache: DynamicCache, target_len: int) -> None:
    cache.crop(target_len)


@torch.inference_mode()
def spec_generate(
    target_lm: LoadedModel,
    draft_lm: LoadedModel,
    prompt: str,
    max_new_tokens: int = 128,
    K: int = 4,
    temperature: float = 1.0,
) -> tuple[str, list[float], float]:
    tok = target_lm.tokenizer
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(target_lm.device)
    eos_ids = set(target_lm.eos_token_ids)

    # Prefill both models.
    target_kv = DynamicCache()
    out = target_lm.model(input_ids=input_ids, past_key_values=target_kv, use_cache=True)
    target_kv = out.past_key_values
    # First token: sample from target at given temperature (greedy at T=0).
    first_logits = out.logits[:, -1, :]
    if temperature == 0.0:
        last_token = first_logits.argmax(dim=-1, keepdim=True)
    else:
        probs = torch.softmax(first_logits / temperature, dim=-1)
        last_token = torch.multinomial(probs, num_samples=1)
    all_ids = torch.cat([input_ids, last_token], dim=1)

    if last_token.item() in eos_ids:
        return tok.decode(all_ids[0], skip_special_tokens=True), [], 0.0

    draft_kv = DynamicCache()
    out = draft_lm.model(input_ids=input_ids, past_key_values=draft_kv, use_cache=True)
    draft_kv = out.past_key_values

    n_generated = 1
    total_accepted = 0
    total_proposed = 0
    per_round_seconds: list[float] = []

    while n_generated < max_new_tokens:
        K_actual = min(K, max_new_tokens - n_generated)
        N = all_ids.shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        draft_tokens, draft_logprobs, draft_kv = run_draft(
            draft_lm, last_token, draft_kv, K_actual, temperature
        )
        target_logits, target_kv = run_verify(target_lm, last_token, draft_tokens, target_kv)

        if temperature == 0.0:
            new_tokens, n_accepted = greedy_accept(draft_tokens, target_logits)
        else:
            new_tokens, n_accepted = rejection_sample(
                draft_tokens, draft_logprobs, target_logits, temperature
            )

        torch.cuda.synchronize()
        per_round_seconds.append(time.perf_counter() - t0)

        total_accepted += n_accepted
        total_proposed += K_actual

        _crop_cache(target_kv, N + n_accepted)
        _crop_cache(draft_kv, N + n_accepted)

        eos_pos = next(
            (i for i, t in enumerate(new_tokens.tolist()) if t in eos_ids), None
        )
        if eos_pos is not None:
            all_ids = torch.cat([all_ids, new_tokens[: eos_pos + 1].unsqueeze(0)], dim=1)
            n_generated += eos_pos + 1
            break

        all_ids = torch.cat([all_ids, new_tokens.unsqueeze(0)], dim=1)
        n_generated += new_tokens.shape[0]
        last_token = new_tokens[-1:].unsqueeze(0)

    text = tok.decode(all_ids[0], skip_special_tokens=True)
    mean_acceptance = total_accepted / total_proposed if total_proposed > 0 else 0.0
    return text, per_round_seconds, mean_acceptance
