import time

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
app = modal.App("tiny-decode-phase3-step5")

MODEL_NAME = "Qwen/Qwen3-4B"
BLOCK_SIZE = 16
MAX_NEW_TOKENS = 256
CONCURRENCY_VALUES = [1, 2, 4, 8, 16, 32]

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
# Remote bench function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_bench_remote(max_running: int, max_new_tokens: int) -> dict:
    """Run DynamicScheduler then PagedScheduler; return throughput + latency stats."""
    import gc
    import torch

    from engine.model import format_prompt, load
    from engine.kv_cache import BlockAllocator
    from engine.scheduler import DynamicScheduler, PagedScheduler

    if "lm" not in _loaded:
        _loaded["lm"] = load(MODEL_NAME)
    lm = _loaded["lm"]

    raw_prompts = mixed_prompts(max_running * 2)
    prompts = [format_prompt(lm.tokenizer, p) for p in raw_prompts]

    cfg = lm.model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    bytes_per_block = (
        cfg.num_hidden_layers * 2 * cfg.num_key_value_heads
        * BLOCK_SIZE * head_dim * 2
    )

    # ---- DynamicScheduler ----
    dyn_oom = False
    dyn_tps = 0.0
    dyn_kv_gib = 0.0
    try:
        model_gib = torch.cuda.memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        sched = DynamicScheduler(lm, prompts, max_new_tokens=max_new_tokens,
                                 max_running=max_running, eos_token_ids=lm.eos_token_ids)
        sched.run()
        wall = time.perf_counter() - t0
        total_tokens = sum(s.all_ids.shape[1] for s in sched.finished)
        dyn_tps = total_tokens / wall
        dyn_kv_gib = torch.cuda.max_memory_allocated() / 1024**3 - model_gib
    except torch.cuda.OutOfMemoryError:
        dyn_oom = True

    gc.collect()
    torch.cuda.empty_cache()

    # ---- PagedScheduler ----
    paged_oom = False
    paged_tps = 0.0
    paged_kv_gib = 0.0
    try:
        n_blocks = BlockAllocator.budget_n_blocks(lm.model, BLOCK_SIZE)
        allocator = BlockAllocator.from_model(lm.model, BLOCK_SIZE, n_blocks, lm.device, lm.dtype)
        t0 = time.perf_counter()
        sched = PagedScheduler(lm, prompts, allocator, max_new_tokens=max_new_tokens,
                               max_running=max_running, eos_token_ids=lm.eos_token_ids)
        sched.run()
        wall = time.perf_counter() - t0
        total_tokens = sum(s.all_ids.shape[1] for s in sched.finished)
        paged_tps = total_tokens / wall
        paged_kv_gib = allocator.peak_allocated * bytes_per_block / 1024**3
        del allocator
    except torch.cuda.OutOfMemoryError:
        paged_oom = True

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "max_running": max_running,
        "dyn_tps": dyn_tps,
        "dyn_kv_gib": dyn_kv_gib,
        "dyn_oom": dyn_oom,
        "paged_tps": paged_tps,
        "paged_kv_gib": paged_kv_gib,
        "paged_oom": paged_oom,
    }


# ---------------------------------------------------------------------------
# tput entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def tput(max_new_tokens: int = MAX_NEW_TOKENS):
    """Total throughput and average per-token latency, dyn vs paged.

    tok/s     = total tokens generated / wall time.  Comparable to
                "total_tok/s" from phase2-step4's sweep table.
    ms/tok    = 1000 / tok/s.  Comparable to "ms/tok (per seq)" from
                phase2-step4 and the Phase 1 single-sequence baseline.
                For sequential forward passes this is flat with concurrency;
                the plateau here mirrors the ~23 tok/s plateau in phase2.
    """
    print()
    print(f"model={MODEL_NAME}  max_new_tokens={max_new_tokens}  block_size={BLOCK_SIZE}")
    print(f"n_prompts = concur × 2  (mixed short + long)")
    print()
    header = (
        f"{'batch_size':>6}  "
        f"{'dyn_tok/s':>10}  {'dyn_ms/tok':>11}  "
        f"{'paged_tok/s':>12}  {'paged_ms/tok':>13}  "
        f"{'dyn_kv':>7}  {'paged_kv':>8}"
    )
    print(header)
    print("-" * len(header))

    for max_running in CONCURRENCY_VALUES:
        out = run_bench_remote.remote(max_running=max_running, max_new_tokens=max_new_tokens)

        if out["dyn_oom"]:
            dyn_tps_s = f"{'(OOM)':>10}"
            dyn_ms_s  = f"{'--':>11}"
            dyn_kv_s  = f"{'--':>7}"
        else:
            dyn_tps_s = f"{out['dyn_tps']:>9.1f}t/s"
            dyn_ms_s  = f"{1000/out['dyn_tps']:>10.1f}ms"
            dyn_kv_s  = f"{out['dyn_kv_gib']:>6.2f}G"

        if out["paged_oom"]:
            paged_tps_s = f"{'(OOM)':>12}"
            paged_ms_s  = f"{'--':>13}"
            paged_kv_s  = f"{'--':>8}"
        else:
            paged_tps_s = f"{out['paged_tps']:>11.1f}t/s"
            paged_ms_s  = f"{1000/out['paged_tps']:>12.1f}ms"
            paged_kv_s  = f"{out['paged_kv_gib']:>7.2f}G"

        print(
            f"{out['max_running']:>6}  "
            f"{dyn_tps_s}  {dyn_ms_s}  "
            f"{paged_tps_s}  {paged_ms_s}  "
            f"{dyn_kv_s}  {paged_kv_s}"
        )
        if out["paged_oom"]:
            break

