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
app = modal.App("tiny-decode-phase3-step1")

MODEL_NAME = "Qwen/Qwen3-4B"
MAX_SEQ_LEN = 2048  # pre-allocation budget per sequence slot

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


def static_kv_gib_formula(model, max_running: int, max_seq_len: int) -> float:
    """Theoretical KV cost if every active slot pre-allocates max_seq_len tokens.

    KV per token = 2 (K + V) × n_kv_heads × head_dim × 2 bytes (fp16)
    Summed over all layers and all concurrent sequence slots.
    """
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_bytes = (
        max_running
        * max_seq_len
        * cfg.num_hidden_layers
        * 2                         # K and V
        * cfg.num_key_value_heads
        * head_dim
        * 2                         # fp16
    )
    return kv_bytes / 1024**3


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_wall_remote(max_running: int, max_seq_len: int) -> dict:
    """Run DynamicScheduler; return measured dynamic KV usage alongside formula-based static cost.

    static_kv_gib:  what a static pre-allocator would commit (max_running × max_seq_len × kv dims)
    dynamic_kv_gib: peak KV overhead actually measured during the DynamicCache run
    """
    import torch

    from engine.model import format_prompt, load
    from engine.scheduler import DynamicScheduler

    if "lm" not in _loaded:
        _loaded["lm"] = load(MODEL_NAME)
    lm = _loaded["lm"]

    model_gib = torch.cuda.memory_allocated() / 1024**3
    prompts = mixed_prompts(max_running * 2)
    formatted = [format_prompt(lm.tokenizer, p) for p in prompts]

    torch.cuda.reset_peak_memory_stats()
    try:
        sched = DynamicScheduler(lm, formatted, max_seq_len, max_running, lm.eos_token_ids)
        sched.run()
    except torch.cuda.OutOfMemoryError:
        return {
            "max_running": max_running,
            "max_seq_len": max_seq_len,
            "model_gib": model_gib,
            "static_kv_gib": static_kv_gib_formula(lm.model, max_running, max_seq_len),
            "dynamic_kv_gib": None,
            "oom": True,
        }

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    dynamic_kv_gib = peak_gib - model_gib
    actual_tokens = [s.eos_decode_step + 1 for s in sched.finished]
    avg_actual = sum(actual_tokens) / len(actual_tokens)

    return {
        "max_running": max_running,
        "max_seq_len": max_seq_len,
        "model_gib": model_gib,
        "static_kv_gib": static_kv_gib_formula(lm.model, max_running, max_seq_len),
        "dynamic_kv_gib": dynamic_kv_gib,
        "avg_actual_tokens": avg_actual,
        "wasted_fraction": 1.0 - avg_actual / max_seq_len,
        "oom": False,
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_waste_remote(max_running: int, max_seq_len: int) -> dict:
    """Run mixed prompts; return per-sequence token usage vs pre-allocation."""
    import torch

    from engine.model import format_prompt, load
    from engine.scheduler import DynamicScheduler

    if "lm" not in _loaded:
        _loaded["lm"] = load(MODEL_NAME)
    lm = _loaded["lm"]

    prompts = mixed_prompts(max_running * 2)
    formatted = [format_prompt(lm.tokenizer, p) for p in prompts]

    sched = DynamicScheduler(lm, formatted, max_seq_len, max_running, lm.eos_token_ids)
    sched.run()

    sched.finished.sort(key=lambda s: s.seq_id)
    rows = [
        {
            "seq_id": s.seq_id,
            "actual_tokens": s.eos_decode_step + 1,
            "allocated_tokens": max_seq_len,
            "utilisation": (s.eos_decode_step + 1) / max_seq_len,
        }
        for s in sched.finished
    ]

    # Static model: each slot holds max_seq_len regardless of actual usage.
    # With continuous batching + 2×max_running prompts, each slot is occupied
    # twice on average, so total allocation = 2 × max_running × max_seq_len.
    total_actual = sum(r["actual_tokens"] for r in rows)
    total_alloc = 2 * max_running * max_seq_len
    return {
        "rows": rows,
        "prompts": prompts,
        "total_actual_tokens": total_actual,
        "total_allocated_tokens": total_alloc,
        "overall_utilisation": total_actual / total_alloc,
        "static_kv_gib": static_kv_gib_formula(lm.model, max_running, max_seq_len),
    }


WALL_MAX_RUNNING_VALUES = [1, 2, 4, 8, 16, 32]


@app.local_entrypoint()
def wall(max_seq_len: int = MAX_SEQ_LEN):
    """Formula-based static KV cost vs measured DynamicCache usage as concurrency grows.

    static_kv:  what a static pre-allocator must commit (max_running × max_seq_len × kv dims)
    dynamic_kv: peak KV overhead actually used by DynamicCache (tracks actual tokens)
    wasted:     fraction of the static budget never written (1 - avg_tok / max_seq_len)

    PagedAttention target: dynamic_kv-level memory with static-level admission control.
    """
    print()
    print(f"model={MODEL_NAME}  max_seq_len={max_seq_len}")
    print()
    header = (
        f"{'concur':>6}  {'static_kv':>10}  {'dynamic_kv':>10}  "
        f"{'avg_tok':>8}  {'wasted':>7}"
    )
    print(header)
    print("-" * len(header))

    for max_running in WALL_MAX_RUNNING_VALUES:
        out = run_wall_remote.remote(max_running=max_running, max_seq_len=max_seq_len)
        if out["oom"]:
            print(
                f"{out['max_running']:>6}  "
                f"{out['static_kv_gib']:>9.2f}G  "
                f"{'(OOM)':>10}  "
            )
            break
        print(
            f"{out['max_running']:>6}  "
            f"{out['static_kv_gib']:>9.2f}G  "
            f"{out['dynamic_kv_gib']:>9.2f}G  "
            f"{out['avg_actual_tokens']:>7.1f}t  "
            f"{out['wasted_fraction']:>6.1%}"
        )


@app.local_entrypoint()
def waste(max_running: int = 4, max_seq_len: int = MAX_SEQ_LEN):
    """Per-sequence KV utilisation on mixed prompts."""
    out = run_waste_remote.remote(max_running=max_running, max_seq_len=max_seq_len)

    prompts = out["prompts"]
    print()
    print(
        f"model={MODEL_NAME}  max_running={max_running}  max_seq_len={max_seq_len}"
    )
    print(
        f"static KV budget: {out['static_kv_gib']:.2f} GiB  "
        f"(for {max_running} concurrent slots × {max_seq_len} tokens)"
    )
    print(
        f"total tokens: actual={out['total_actual_tokens']}  "
        f"allocated={out['total_allocated_tokens']}  "
        f"utilisation={out['overall_utilisation']:.1%}"
    )
    print()
    header = f"{'seq':>4}  {'prompt':<40}  {'actual':>7}  {'alloc':>6}  {'util':>6}"
    print(header)
    print("-" * len(header))
    for r in out["rows"]:
        short = prompts[r["seq_id"]]
        short = short if len(short) <= 38 else short[:35] + "..."
        print(
            f"{r['seq_id']:>4}  {short:<40}  "
            f"{r['actual_tokens']:>6}t  "
            f"{r['allocated_tokens']:>5}t  "
            f"{r['utilisation']:>5.1%}"
        )
    print()
    wasted = 1.0 - out["overall_utilisation"]
    print(
        f"  {wasted:.1%} of pre-allocated KV was never written — "
        f"PagedAttention eliminates this waste."
    )
