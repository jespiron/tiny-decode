from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    name: str
    # All token IDs that signal end-of-sequence for this model. May include
    # multiple IDs when the model uses different stop tokens in chat format
    # vs. plain text (e.g. Qwen3: <|endoftext|> and <|im_end|>).
    eos_token_ids: list[int] = field(default_factory=list)


def load(name: str = "Qwen/Qwen3-4B") -> LoadedModel:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Run via Modal.")
    device = torch.device("cuda")
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect all stop tokens. Qwen3 instruction models end responses with
    # <|im_end|> (151645) rather than the document-level <|endoftext|>
    # (151643). We include both so generate_batch catches either.
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        eos_ids.add(im_end)

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        name=name,
        eos_token_ids=sorted(eos_ids),
    )


def format_prompt(tokenizer, prompt: str, enable_thinking: bool = False) -> str:
    # Note to self: this basically wraps a plain prompt in the model's instruction-turn format.
    #
    # Without this magical piece of scaffolding, instruction-tuned models run in raw continuation mode and
    # never generate a natural stop token :/ 
    #
    # With the chat template applied, the model generates a response and ends
    # it with <|im_end|>, which load() includes in eos_token_ids
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
