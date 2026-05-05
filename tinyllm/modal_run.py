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
app = modal.App("tiny-decode-phase2-step3")

MODEL_NAME = "Qwen/Qwen3-4B"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_remote(
    prompts: list[str], max_new_tokens: int, max_running: int, warmup: int, runs: int
) -> dict:
    from engine.decode import generate_batch
    from engine.model import load
    from harness.bench import run_batch_benchmark, to_dict

    lm = load(MODEL_NAME)

    def gen():
        return generate_batch(
            lm, prompts, max_new_tokens=max_new_tokens,
            max_running=max_running, eos_token_ids=lm.eos_token_ids,
        )

    # batch_size=len(prompts) overcounts tokens_per_sec for continuous batching:
    # each step() advances at most max_running sequences, not len(prompts).
    # Precise counting requires eos_decode_step from generate_batch's third
    # return value; deferred to a dedicated benchmark in a later phase.
    result = run_batch_benchmark(
        generate_fn=gen,
        model_name=lm.name,
        batch_size=len(prompts),
        warmup=warmup,
        runs=runs,
    )
    texts, _, _ = gen()
    return {"texts": texts, "result": to_dict(result)}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_latency_remote(
    prompts: list[str], max_new_tokens: int, max_running: int
) -> dict:
    from engine.model import format_prompt, load
    from engine.scheduler import Scheduler

    lm = load(MODEL_NAME)
    formatted = [format_prompt(lm.tokenizer, p) for p in prompts]

    sched = Scheduler(
        lm, formatted,
        max_new_tokens=max_new_tokens,
        max_running=max_running,
        eos_token_ids=lm.eos_token_ids,
    )
    sched.run()

    t0 = min(s.submitted_at for s in sched.finished)
    rows = []
    for s in sched.finished:
        response = s.text.rsplit("\nassistant\n", 1)[-1] if "\nassistant\n" in s.text else s.text
        rows.append({
            "seq_id": s.seq_id,
            "started_ms": (s.started_at - t0) * 1000,
            "finished_ms": (s.finished_at - t0) * 1000,
            "latency_ms": (s.finished_at - s.submitted_at) * 1000,
            "eos_decode_step": s.eos_decode_step,
            "preview": response[:80].replace("\n", " ").strip(),
        })

    return {"rows": rows, "total_ms": (max(s.finished_at for s in sched.finished) - t0) * 1000}


@app.local_entrypoint()
def main(
    prompt: str = "The transformer architecture revolutionized NLP because",
    max_new_tokens: int = 128,
    warmup: int = 1,
    runs: int = 3,
    check: bool = False,
):
    import sys

    out = run_remote.remote(
        prompts=[prompt], max_new_tokens=max_new_tokens,
        max_running=1, warmup=warmup, runs=runs,
    )
    r = out["result"]

    if check:
        from harness.snapshot import check_or_write

        passed, msg = check_or_write(
            model=MODEL_NAME,
            prompt=prompt,
            n_tokens=max_new_tokens,
            text=out["texts"][0],
            write=False,
        )
        print(f"\n[snapshot] {msg}")
        if not passed:
            sys.exit(1)

    print(out["texts"][0])
    print()
    print(f"model={r['model']}  batch_size={r['batch_size']}  runs={r['runs']}")
    print(f"  tokens/sec:  {r['tokens_per_sec']:.1f}  (total across batch)")
    print(f"  ms/token:    {r['ms_per_token']:.2f}  (per sequence, decode)")
    print(f"  TTFT:        {r['ttft_ms']:.1f} ms")
    print(f"  peak memory: {r['peak_memory_mb']:.0f} MB")


# Mixed prompts: alternating short (factual) and long (generative) to make
# the latency difference between static and continuous batching visible.
LATENCY_DEMO_PROMPTS = [
    "What is the capital of France?",
    "Explain in detail how the attention mechanism works in a transformer.",
    "Is Python interpreted or compiled? Answer in one sentence.",
    "Write a short story about a robot who learns to paint.",
    "What is 2 + 2?",
    "Describe the history of the Roman Empire in significant detail.",
    "Name one fruit.",
    "What are the key differences between supervised and unsupervised learning?",
]
LATENCY_DEMO_MAX_RUNNING = 4
LATENCY_DEMO_MAX_TOKENS = 200


@app.local_entrypoint()
def latency_demo(
    max_running: int = LATENCY_DEMO_MAX_RUNNING,
    max_new_tokens: int = LATENCY_DEMO_MAX_TOKENS,
):
    prompts = LATENCY_DEMO_PROMPTS
    out = run_latency_remote.remote(
        prompts=prompts, max_new_tokens=max_new_tokens, max_running=max_running
    )
    rows = out["rows"]
    total_ms = out["total_ms"]

    print()
    print(f"model={MODEL_NAME}  max_new_tokens={max_new_tokens}  max_running={max_running}")
    print(f"total prompts={len(prompts)}  total wall time={total_ms:.0f} ms")
    print()

    header = f"{'seq':>4}  {'started':>8}  {'finished':>9}  {'latency':>9}  {'steps':>6}  preview"
    print(header)
    print("-" * len(header))
    for r in rows:
        short = LATENCY_DEMO_PROMPTS[r["seq_id"]]
        short = short if len(short) <= 38 else short[:35] + "..."
        print(
            f"{r['seq_id']:>4}  "
            f"{r['started_ms']:>7.0f}ms  "
            f"{r['finished_ms']:>8.0f}ms  "
            f"{r['latency_ms']:>8.0f}ms  "
            f"{r['eos_decode_step']:>6}  "
            f"{r['preview'][:40]!r}"
        )
    print()
    print("Short requests (seq 0, 2, 4, 6) finish early and free slots for the next waiting")
    print("sequence. With static batching they would have waited for the batch's slowest sequence.")
