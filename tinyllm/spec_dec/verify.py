import torch
from transformers import DynamicCache

from engine.model import LoadedModel


@torch.inference_mode()
def run_verify(
    target_lm: LoadedModel,
    last_token: torch.Tensor,    # [1, 1]
    draft_tokens: torch.Tensor,  # [K]
    kv: DynamicCache,
) -> tuple[torch.Tensor, DynamicCache]:
    draft_ids = draft_tokens.unsqueeze(0)                          # [1, K]
    input_ids = torch.cat([last_token, draft_ids], dim=1)          # [1, K+1]
    out = target_lm.model(input_ids=input_ids, past_key_values=kv, use_cache=True)
    return out.logits[0], out.past_key_values                      # [K+1, vocab]
