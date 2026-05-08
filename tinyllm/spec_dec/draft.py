import torch
from transformers import DynamicCache

from engine.model import LoadedModel


@torch.inference_mode()
def run_draft(
    draft_lm: LoadedModel,
    last_token: torch.Tensor,  # [1, 1]
    kv: DynamicCache,
    K: int,
) -> tuple[torch.Tensor, DynamicCache]:
    tokens = []
    current = last_token
    for _ in range(K):
        out = draft_lm.model(input_ids=current, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        current = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        tokens.append(current)
    return torch.cat(tokens, dim=1).squeeze(0), kv  # [K]
