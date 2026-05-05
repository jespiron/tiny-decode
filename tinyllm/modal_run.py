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
app = modal.App("tiny-decode-phase2-step4")

MODEL_NAME = "Qwen/Qwen3-4B"

# Module-level cache so warm container reuse doesn't reload the model.
# Modal keeps containers alive between calls; without this each invocation
# would load a fresh copy on top of the still-live previous one.
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

# interleave short and long prompts, cycling as needed.
# ensures that every batch contains both short and long sequences
def mixed_prompts(n: int) -> list[str]:
    pool = [p for pair in zip(SHORT_PROMPTS, LONG_PROMPTS) for p in pair]
    return [pool[i % len(pool)] for i in range(n)]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_throughput_remote(
    batch_size: int, max_new_tokens: int, warmup: int, runs: int
) -> dict:
    import gc

    import torch
    from engine.decode import generate_batch as static_gen_batch
    from engine.scheduler import Scheduler
    from harness.bench import run_batch_benchmark, to_dict

    if "lm" not in _loaded:
        from engine.model import load
        _loaded["lm"] = load(MODEL_NAME)
    lm = _loaded["lm"]

    prompt = "The transformer architecture revolutionized NLP because"
    prompts = [prompt] * batch_size

    # --- Static ---
    def static_gen():
        return static_gen_batch(lm, prompts, max_new_tokens, lm.eos_token_ids)

    static_result = run_batch_benchmark(
        generate_fn=static_gen,
        model_name=lm.name,
        batch_size=batch_size,
        warmup=warmup,
        runs=runs,
    )

    gc.collect()
    torch.cuda.empty_cache()

    # --- Continuous (max_running = batch_size, same prompts, same concurrency) ---
    # batch_size=len(prompts) overcounts tok/s for continuous; noted here and
    # addressed in a later phase with precise per-sequence accounting.
    def continuous_gen():
        return Scheduler(lm, prompts, max_new_tokens, batch_size, lm.eos_token_ids).run()

    continuous_result = run_batch_benchmark(
        generate_fn=continuous_gen,
        model_name=lm.name,
        batch_size=batch_size,
        warmup=warmup,
        runs=runs,
    )

    return {"static": to_dict(static_result), "continuous": to_dict(continuous_result)}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_latency_remote(max_new_tokens: int, max_running: int) -> dict:
    import time

    import torch
    from engine.decode import generate_batch as static_gen_batch
    from engine.model import format_prompt
    from engine.scheduler import Scheduler

    if "lm" not in _loaded:
        from engine.model import load
        _loaded["lm"] = load(MODEL_NAME)
    lm = _loaded["lm"]
    print(f"[latency] model mem={torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    prompts = mixed_prompts(max_running * 2)
    formatted = [format_prompt(lm.tokenizer, p) for p in prompts]
    n = len(formatted)

    # --- Static: process in sequential batches of max_running ---
    static_rows = []
    cumulative_ms = 0.0
    for start in range(0, n, max_running):
        batch = formatted[start : start + max_running]
        _, per_step, eos_steps = static_gen_batch(lm, batch, max_new_tokens, lm.eos_token_ids)
        batch_ms = sum(per_step) * 1000
        cumulative_ms += batch_ms
        for i, eos in enumerate(eos_steps):
            static_rows.append(
                {"seq_id": start + i, "latency_ms": cumulative_ms, "eos_decode_step": eos}
            )

    # --- Continuous ---
    sched = Scheduler(lm, formatted, max_new_tokens, max_running, lm.eos_token_ids)
    sched.run()
    t0 = min(s.submitted_at for s in sched.finished)
    sched.finished.sort(key=lambda s: s.seq_id)
    continuous_rows = [
        {
            "seq_id": s.seq_id,
            "latency_ms": (s.finished_at - t0) * 1000,
            "started_ms": (s.started_at - t0) * 1000,
            "eos_decode_step": s.eos_decode_step,
        }
        for s in sched.finished
    ]

    return {
        "prompts": prompts,
        "static_rows": static_rows,
        "continuous_rows": continuous_rows,
        "static_total_ms": cumulative_ms,
        "continuous_total_ms": (max(s.finished_at for s in sched.finished) - t0) * 1000,
    }


SWEEP_BATCH_SIZES = [1, 2, 4, 8, 16]
SWEEP_NEW_TOKENS = 64  # 256 gets OOM'd at batch >= 16 on an A10G (24 GB) with Qwen3-4B........


@app.local_entrypoint()
def sweep(warmup: int = 1, runs: int = 3):
    rows = []
    for bs in SWEEP_BATCH_SIZES:
        out = run_throughput_remote.remote(
            batch_size=bs, max_new_tokens=SWEEP_NEW_TOKENS, warmup=warmup, runs=runs
        )
        rows.append((bs, out["static"], out["continuous"]))

    print()
    print(f"model={MODEL_NAME}  new_tokens={SWEEP_NEW_TOKENS}  runs={runs}")
    print()
    print(
        "| batch_size | static tok/s | continuous tok/s "
        "| static ms/tok | continuous ms/tok |"
    )
    print("|---:|---:|---:|---:|---:|")
    for bs, s, c in rows:
        print(
            f"| {bs} "
            f"| {s['tokens_per_sec']:.1f} | {c['tokens_per_sec']:.1f} "
            f"| {s['ms_per_token']:.2f} | {c['ms_per_token']:.2f} |"
        )


LATENCY_DEMO_MAX_RUNNING = 4
LATENCY_DEMO_MAX_TOKENS = 128


@app.local_entrypoint()
def latency_demo(
    max_running: int = LATENCY_DEMO_MAX_RUNNING,
    max_new_tokens: int = LATENCY_DEMO_MAX_TOKENS,
):
    out = run_latency_remote.remote(
        max_new_tokens=max_new_tokens, max_running=max_running
    )

    prompts = out["prompts"]
    static_rows = {r["seq_id"]: r for r in out["static_rows"]}
    continuous_rows = {r["seq_id"]: r for r in out["continuous_rows"]}

    print()
    print(
        f"model={MODEL_NAME}  max_new_tokens={max_new_tokens}  "
        f"max_running={max_running}  total_prompts={len(prompts)}"
    )
    print(
        f"static total={out['static_total_ms']:.0f} ms  "
        f"continuous total={out['continuous_total_ms']:.0f} ms"
    )
    print()

    header = f"{'seq':>4}  {'prompt':<38}  {'static ms':>10}  {'cont ms':>10}  {'steps':>6}  {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for i, prompt in enumerate(prompts):
        s = static_rows[i]
        c = continuous_rows[i]
        short = prompt if len(prompt) <= 36 else prompt[:33] + "..."
        speedup = s["latency_ms"] / c["latency_ms"] if c["latency_ms"] > 0 else 0
        print(
            f"{i:>4}  {short:<38}  "
            f"{s['latency_ms']:>9.0f}ms  "
            f"{c['latency_ms']:>9.0f}ms  "
            f"{c['eos_decode_step']:>6}  "
            f"{speedup:>7.1f}×"
        )

    print()
    short_ids = [i for i in range(len(prompts)) if i % 2 == 0]
    avg_static = sum(static_rows[i]["latency_ms"] for i in short_ids) / len(short_ids)
    avg_cont = sum(continuous_rows[i]["latency_ms"] for i in short_ids) / len(short_ids)
    print(
        f"short-request avg latency:  static={avg_static:.0f} ms  "
        f"continuous={avg_cont:.0f} ms  ({avg_static/avg_cont:.1f}× improvement)"
    )
