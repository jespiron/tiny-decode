import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.6.2",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env(
        {
            "HF_HOME": "/cache",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "HF_HUB_VERBOSITY": "error",
            "PYTHONWARNINGS": "ignore",
        }
    )
    .add_local_python_source("engine", "harness")
)

hf_cache = modal.Volume.from_name("tiny-decode-hf-cache", create_if_missing=True)
app = modal.App("tiny-decode-phase3-step4")

MODEL_NAME = "Qwen/Qwen3-4B"
BLOCK_SIZE = 16

_loaded: dict = {}

SHORT_PROMPTS = [
    "What is the capital of France?",
    "Is Python interpreted or compiled? Answer in one sentence.",
    "What is 2 + 2?",
    "Name one fruit.",
]
LONG_PROMPTS = [
    "Explain in detail how the attention mechanism works in a transformer.",
    "Write a short story about a robot who learns to paint.",
    "Describe the history of the Roman Empire in significant detail.",
    "What are the key differences between supervised and unsupervised learning?",
]


def mixed_prompts(n: int) -> list[str]:
    pool = [p for pair in zip(SHORT_PROMPTS, LONG_PROMPTS) for p in pair]
    return [pool[i % len(pool)] for i in range(n)]


# ---------------------------------------------------------------------------
# Snapshot / correctness check
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS_SNAPSHOT = 64
ALL_PROMPTS = SHORT_PROMPTS + LONG_PROMPTS


def _dynamic_run(lm, formatted: str, max_new_tokens: int) -> str:
    """Run a single prompt with inline DynamicCache; return decoded text."""
    import torch
    from transformers import DynamicCache

    tok = lm.tokenizer
    enc = tok([formatted], return_tensors="pt", padding=False)
    input_ids = enc.input_ids.to(lm.device)
    attn_mask = enc.attention_mask.to(lm.device)
    eos_ids = torch.tensor(lm.eos_token_ids, device=lm.device)

    kv = DynamicCache()
    with torch.inference_mode():
        out = lm.model(input_ids=input_ids, attention_mask=attn_mask,
                       past_key_values=kv, use_cache=True)
    kv = out.past_key_values
    all_ids = torch.cat([input_ids, out.logits[:, -1:, :].argmax(-1)], dim=1)

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            out = lm.model(
                input_ids=all_ids[:, -1:],
                attention_mask=torch.ones(1, all_ids.shape[1], device=lm.device),
                past_key_values=kv, use_cache=True,
            )
        kv = out.past_key_values
        next_id = out.logits[:, -1:, :].argmax(-1)
        all_ids = torch.cat([all_ids, next_id], dim=1)
        if torch.isin(next_id.squeeze(), eos_ids).item():
            break
    return tok.decode(all_ids[0], skip_special_tokens=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=600,
)
def run_snapshot() -> dict:
    """Run every prompt through both schedulers; compare output text for each."""
    from engine.model import format_prompt, load
    from engine.kv_cache import BlockAllocator
    from engine.scheduler import PagedScheduler

    if "lm" not in _loaded:
        _loaded["lm"] = load(MODEL_NAME)
    lm = _loaded["lm"]

    n_blocks = BlockAllocator.budget_n_blocks(lm.model, BLOCK_SIZE)
    allocator = BlockAllocator.from_model(lm.model, BLOCK_SIZE, n_blocks, lm.device, lm.dtype)

    results = []
    for prompt in ALL_PROMPTS:
        formatted = format_prompt(lm.tokenizer, prompt)

        dynamic_text = _dynamic_run(lm, formatted, MAX_NEW_TOKENS_SNAPSHOT)

        sched = PagedScheduler(lm, [formatted], allocator,
                               max_new_tokens=MAX_NEW_TOKENS_SNAPSHOT,
                               max_running=1, eos_token_ids=lm.eos_token_ids)
        texts, _, _ = sched.run()
        paged_text = texts[0]

        results.append({
            "prompt": prompt,
            "match": dynamic_text == paged_text,
            "dynamic_text": dynamic_text[:120],
            "paged_text": paged_text[:120],
        })

    return {
        "results": results,
        "all_match": all(r["match"] for r in results),
    }


@app.local_entrypoint()
def snapshot():
    out = run_snapshot.remote()
    print()
    all_match = out["all_match"]
    print(f"all prompts match: {all_match}")
    print()
    for r in out["results"]:
        status = "OK " if r["match"] else "FAIL"
        print(f"  [{status}] {r['prompt'][:60]!r}")
        if not r["match"]:
            print(f"         dynamic: {r['dynamic_text']!r}")
            print(f"          paged:  {r['paged_text']!r}")
    if not all_match:
        print("\nWARNING: some prompts differ — paged attention has a bug.")

