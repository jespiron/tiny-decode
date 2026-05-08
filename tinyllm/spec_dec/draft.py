import torch
import torch.nn.functional as F
from transformers import DynamicCache

from engine.model import LoadedModel


@torch.inference_mode()
def run_draft(
    draft_lm: LoadedModel,
    last_token: torch.Tensor,  # [1, 1]
    kv: DynamicCache,
    K: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, DynamicCache]:
    tokens: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    current = last_token

    for _ in range(K):
        out = draft_lm.model(input_ids=current, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        raw_logits = out.logits[:, -1, :]  # [1, vocab]

        if temperature == 0.0:
            current = raw_logits.argmax(dim=-1, keepdim=True)
            lp = F.log_softmax(raw_logits, dim=-1)
        else:
            scaled = raw_logits / temperature
            lp = F.log_softmax(scaled, dim=-1)
            current = torch.multinomial(lp.exp(), num_samples=1)  # [1, 1]

        tokens.append(current)
        logprobs.append(lp)

    draft_tokens = torch.cat(tokens, dim=1).squeeze(0)       # [K]
    draft_logprobs = torch.cat(logprobs, dim=0)              # [K, vocab]
    return draft_tokens, draft_logprobs, kv
