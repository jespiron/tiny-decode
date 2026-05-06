from math import ceil

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
app = modal.App("tiny-decode-phase3-step2")

MODEL_NAME = "Qwen/Qwen3-4B"
BLOCK_SIZE = 16
MAX_SEQ_LEN = 2048

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


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_alloc_demo(max_running: int, max_seq_len: int, block_size: int) -> dict:
    """Run DynamicScheduler and report per-sequence block usage."""
    import torch

    from engine.model import format_prompt, load
    from engine.scheduler import DynamicScheduler
    from engine.kv_cache import BlockAllocator

    if "lm" not in _loaded:
        _loaded["lm"] = load(MODEL_NAME)
    lm = _loaded["lm"]

    prompts = mixed_prompts(max_running * 2)
    formatted = [format_prompt(lm.tokenizer, p) for p in prompts]

    sched = DynamicScheduler(lm, formatted, max_seq_len, max_running, lm.eos_token_ids)
    sched.run()
    sched.finished.sort(key=lambda s: s.seq_id)

    # Total KV tokens per sequence (prompt + generated).
    total_tokens = [s.all_ids.shape[1] for s in sched.finished]
    prompts_used = [s.prompt for s in sched.finished]

    # How many blocks each sequence would have needed with a block allocator.
    blocks_used = [ceil(t / block_size) for t in total_tokens]
    # How many blocks static pre-allocation reserves per slot.
    blocks_static = ceil(max_seq_len / block_size)

    # Peak concurrent blocks: at any moment up to max_running sequences are
    # active. The peak happens when all active slots are at their busiest.
    # We approximate: mean blocks per active sequence × max_running.
    avg_blocks = sum(blocks_used) / len(blocks_used)
    peak_paged_approx = round(avg_blocks * max_running)
    peak_static = blocks_static * max_running

    n_blocks = BlockAllocator.budget_n_blocks(lm.model, block_size)

    return {
        "max_running": max_running,
        "max_seq_len": max_seq_len,
        "block_size": block_size,
        "n_blocks_available": n_blocks,
        "blocks_static_per_slot": blocks_static,
        "peak_static": peak_static,
        "peak_paged_approx": peak_paged_approx,
        "avg_blocks_per_seq": avg_blocks,
        "sequences": [
            {
                "seq_id": s.seq_id,
                "prompt": prompts_used[i][:50],
                "total_tokens": total_tokens[i],
                "blocks_used": blocks_used[i],
                "blocks_static": blocks_static,
                "waste_pct": 1.0 - blocks_used[i] / blocks_static,
            }
            for i, s in enumerate(sched.finished)
        ],
    }


@app.local_entrypoint()
def alloc_demo(
    max_running: int = 8,
    max_seq_len: int = MAX_SEQ_LEN,
    block_size: int = BLOCK_SIZE,
):
    """Show per-sequence block usage vs static worst-case."""
    out = run_alloc_demo.remote(
        max_running=max_running,
        max_seq_len=max_seq_len,
        block_size=block_size,
    )

    print()
    print(
        f"model={MODEL_NAME}  block_size={block_size}  "
        f"max_seq_len={max_seq_len}  max_running={max_running}"
    )
    print(f"pool capacity: {out['n_blocks_available']} blocks available after model load")
    print()

    header = f"{'seq':>4}  {'prompt':<42}  {'tokens':>7}  {'blocks':>7}  {'static':>7}  {'waste':>6}"
    print(header)
    print("-" * len(header))
    for row in out["sequences"]:
        print(
            f"{row['seq_id']:>4}  {row['prompt']:<42}  "
            f"{row['total_tokens']:>6}t  "
            f"{row['blocks_used']:>6}b  "
            f"{row['blocks_static']:>6}b  "
            f"{row['waste_pct']:>5.1%}"
        )

    print()
    print(f"  static pre-allocation:  {out['peak_static']} blocks  "
          f"({out['max_running']} slots × {out['blocks_static_per_slot']} blocks/slot)")
    print(f"  paged (approx peak):    {out['peak_paged_approx']} blocks  "
          f"(avg {out['avg_blocks_per_seq']:.1f} blocks/seq × {out['max_running']} concurrent)")
    print(f"  reduction:              {out['peak_static'] / out['peak_paged_approx']:.1f}×")
