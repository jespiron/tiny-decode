import torch

from .model import LoadedModel
from .scheduler import Scheduler

@torch.inference_mode()
def generate(
    lm: LoadedModel, prompt: str, max_new_tokens: int = 64
) -> tuple[str, list[float]]:
    texts, per_step, _ = generate_batch(lm, [prompt], max_new_tokens, max_running=1)
    return texts[0], per_step


@torch.inference_mode()
def generate_batch(
    lm: LoadedModel,
    prompts: list[str],
    max_new_tokens: int = 64,
    max_running: int = 8,
    eos_token_ids: list[int] | None = None,
) -> tuple[list[str], list[float], list[int]]:
    return Scheduler(lm, prompts, max_new_tokens, max_running, eos_token_ids).run()
