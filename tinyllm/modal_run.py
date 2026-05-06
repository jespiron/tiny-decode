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
app = modal.App("tiny-decode-phase3-step3")

MODEL_NAME = "Qwen/Qwen3-4B"
BLOCK_SIZE = 16
MAX_NEW_TOKENS = 60   # enough to cross a few block boundaries

_loaded: dict = {}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=600,
)
def run_table_demo(prompt: str, max_new_tokens: int, block_size: int) -> dict:
    """Run one sequence; record block table snapshots at each block boundary."""
    import torch
    from transformers import DynamicCache

    from engine.model import format_prompt, load
    from engine.kv_cache import BlockAllocator, BlockTable

    if "lm" not in _loaded:
        _loaded["lm"] = load(MODEL_NAME)
    lm = _loaded["lm"]

    tok = lm.tokenizer
    formatted = format_prompt(tok, prompt)
    enc = tok([formatted], return_tensors="pt", padding=False)
    input_ids = enc.input_ids.to(lm.device)
    attn_mask = enc.attention_mask.to(lm.device)
    n_prompt = input_ids.shape[1]

    # We build a BlockTable alongside the real DynamicCache.
    # Allocator uses CPU (no GPU memory needed for the demo — we're only
    # tracking the mapping, not storing KV data in it).
    allocator = BlockAllocator(
        n_blocks=256,
        n_layers=lm.model.config.num_hidden_layers,
        n_kv_heads=lm.model.config.num_key_value_heads,
        block_size=block_size,
        head_dim=lm.model.config.hidden_size // lm.model.config.num_attention_heads,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    table = BlockTable(block_size=block_size)
    snapshots: list[dict] = []

    def snapshot(label: str) -> None:
        snapshots.append({
            "label": label,
            "n_filled": table.n_filled,
            "blocks": table.describe(),
        })

    def advance_table(n_new_tokens: int) -> None:
        for _ in range(n_new_tokens):
            table.maybe_extend(allocator)
            table.record_token()
            # Snapshot at every block boundary.
            if table.n_filled % block_size == 0:
                snapshot(f"after token {table.n_filled}")

    # Prefill: n_prompt tokens enter the KV cache.
    with torch.inference_mode():
        out = lm.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
    kv = out.past_key_values
    advance_table(n_prompt)
    snapshot(f"after prefill ({n_prompt} prompt tokens)")

    # Decode: one token at a time.
    eos_ids = torch.tensor(lm.eos_token_ids, device=lm.device)
    all_ids = torch.cat([input_ids, out.logits[:, -1:, :].argmax(-1)], dim=1)

    for step in range(max_new_tokens):
        with torch.inference_mode():
            out = lm.model(
                input_ids=all_ids[:, -1:],
                attention_mask=torch.ones(1, all_ids.shape[1], device=lm.device),
                past_key_values=kv,
                use_cache=True,
            )
        kv = out.past_key_values
        next_id = out.logits[:, -1:, :].argmax(-1)
        all_ids = torch.cat([all_ids, next_id], dim=1)
        advance_table(1)

        if torch.isin(next_id.squeeze(), eos_ids).item():
            break

    return {
        "prompt": prompt,
        "n_prompt_tokens": n_prompt,
        "n_generated": all_ids.shape[1] - n_prompt,
        "total_tokens": all_ids.shape[1],
        "block_size": block_size,
        "snapshots": snapshots,
        "final_blocks": table.describe(),
    }


@app.local_entrypoint()
def table_demo(
    block_size: int = BLOCK_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    prompt = "Explain how the attention mechanism works in a transformer model."
    out = run_table_demo.remote(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
    )

    print()
    print(f"prompt: \"{out['prompt']}\"")
    print(
        f"prompt tokens: {out['n_prompt_tokens']}  "
        f"generated: {out['n_generated']}  "
        f"total: {out['total_tokens']}  "
        f"block_size: {out['block_size']}"
    )

    for snap in out["snapshots"]:
        print()
        print(f"  [{snap['label']}]  ({snap['n_filled']} tokens in KV cache)")
        print(f"  {'logical':>8}  {'physical':>9}  {'tokens':>14}  {'fill':>8}")
        for b in snap["blocks"]:
            print(
                f"  {b['logical']:>8}  {b['physical']:>9}  "
                f"  {b['token_range']:>12}  "
                f"  {b['filled']}/{b['capacity']}"
            )

    print()
    print("final block table:")
    print(f"  {'logical':>8}  {'physical':>9}  {'tokens':>14}  {'fill':>8}")
    for b in out["final_blocks"]:
        print(
            f"  {b['logical']:>8}  {b['physical']:>9}  "
            f"  {b['token_range']:>12}  "
            f"  {b['filled']}/{b['capacity']}"
        )
