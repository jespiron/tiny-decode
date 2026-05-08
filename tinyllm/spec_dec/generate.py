import time

import torch
from transformers import DynamicCache

from engine.model import LoadedModel
from .draft import run_draft
from .sampler import greedy_accept
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
) -> tuple[str, list[float], float]:
    tok = target_lm.tokenizer
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(target_lm.device)
    eos_ids = set(target_lm.eos_token_ids)

    # --- Prefill both models with the prompt ---
    # The target's first generated token drives what the draft model aligns to;
    # draft prefill just builds its KV state from the same prompt context.
    target_kv = DynamicCache()
    out = target_lm.model(input_ids=input_ids, past_key_values=target_kv, use_cache=True)
    target_kv = out.past_key_values
    last_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
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

    # --- Speculation loop ---
    while n_generated < max_new_tokens:
        K_actual = min(K, max_new_tokens - n_generated)
        # N = sequence length before this round; KV caches each have N-1 entries.
        N = all_ids.shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        draft_tokens, draft_kv = run_draft(draft_lm, last_token, draft_kv, K_actual)
        target_logits, target_kv = run_verify(target_lm, last_token, draft_tokens, target_kv)
        new_tokens, n_accepted = greedy_accept(draft_tokens, target_logits)

        torch.cuda.synchronize()
        per_round_seconds.append(time.perf_counter() - t0)

        total_accepted += n_accepted
        total_proposed += K_actual

        # Roll both caches back to the last accepted position.
        # After verify: target_kv has N+K entries, draft_kv has N+K-1 entries.
        # We keep only N+n_accepted entries (positions 0..N+n_accepted-1).
        _crop_cache(target_kv, N + n_accepted)
        _crop_cache(draft_kv, N + n_accepted)

        # Find the first EOS in new_tokens (if any) and stop there.
        eos_pos = next(
            (i for i, t in enumerate(new_tokens.tolist()) if t in eos_ids), None
        )
        if eos_pos is not None:
            all_ids = torch.cat([all_ids, new_tokens[: eos_pos + 1].unsqueeze(0)], dim=1)
            n_generated += eos_pos + 1
            break

        all_ids = torch.cat([all_ids, new_tokens.unsqueeze(0)], dim=1)
        n_generated += new_tokens.shape[0]
        last_token = new_tokens[-1:].unsqueeze(0)  # [1, 1] — correction or bonus

    text = tok.decode(all_ids[0], skip_special_tokens=True)
    mean_acceptance = total_accepted / total_proposed if total_proposed > 0 else 0.0
    return text, per_round_seconds, mean_acceptance
