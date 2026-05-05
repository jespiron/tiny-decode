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
app = modal.App("tiny-decode-phase2-step1")

MODEL_NAME = "Qwen/Qwen3-4B"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_remote(prompts: list[str], max_new_tokens: int, warmup: int, runs: int) -> dict:
    from engine.decode import generate_batch
    from engine.model import load
    from harness.bench import run_batch_benchmark, to_dict

    lm = load(MODEL_NAME)

    def gen():
        return generate_batch(lm, prompts, max_new_tokens=max_new_tokens,
                              eos_token_ids=lm.eos_token_ids)

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
def run_waste_remote(prompts: list[str], max_new_tokens: int) -> dict:
    from engine.decode import generate_batch
    from engine.model import format_prompt, load

    lm = load(MODEL_NAME)

    # format_prompt applies the instruction-turn template so the model gives
    # a real answer and stops naturally. See engine/model.py for details.
    formatted = [format_prompt(lm.tokenizer, p) for p in prompts]

    texts, per_step, eos_decode_step = generate_batch(
        lm, formatted, max_new_tokens=max_new_tokens,
        eos_token_ids=lm.eos_token_ids,
    )

    # extract just the assistant response for display
    previews = []
    for text in texts:
        response = text.rsplit("\nassistant\n", 1)[-1] if "\nassistant\n" in text else text
        previews.append(response[:80].replace("\n", " ").strip())

    return {
        "previews": previews,
        "eos_decode_step": eos_decode_step,
        "total_decode_steps": len(per_step) - 1,
    }


@app.local_entrypoint()
def main(
    prompt: str = "The transformer architecture revolutionized NLP because",
    max_new_tokens: int = 128,
    batch_size: int = 1,
    warmup: int = 1,
    runs: int = 3,
    check: bool = False,
):
    import sys

    prompts = [prompt] * batch_size
    out = run_remote.remote(
        prompts=prompts, max_new_tokens=max_new_tokens, warmup=warmup, runs=runs
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


SWEEP_PROMPT = "The transformer architecture revolutionized NLP because"
SWEEP_NEW_TOKENS = 128
SWEEP_BATCH_SIZES = [1, 2, 4, 8, 16]


@app.local_entrypoint()
def sweep(warmup: int = 1, runs: int = 3):
    rows = []
    for bs in SWEEP_BATCH_SIZES:
        prompts = [SWEEP_PROMPT] * bs
        out = run_remote.remote(
            prompts=prompts,
            max_new_tokens=SWEEP_NEW_TOKENS,
            warmup=warmup,
            runs=runs,
        )
        rows.append(out["result"])

    print()
    print(f"model={MODEL_NAME}  new_tokens={SWEEP_NEW_TOKENS}  runs={runs}")
    print()
    print("| batch_size | tok/s (total) | ms/tok (per seq) | TTFT (ms) | Peak MB |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['batch_size']} | {r['tokens_per_sec']:.1f} "
            f"| {r['ms_per_token']:.2f} | {r['ttft_ms']:.1f} "
            f"| {r['peak_memory_mb']:.0f} |"
        )


WASTE_DEMO_PROMPTS = [
    "What is the capital of France?",
    "Explain in detail how the attention mechanism works in a transformer.",
    "Is Python interpreted or compiled? Answer in one sentence.",
    "Write a short story about a robot who learns to paint.",
]
WASTE_DEMO_MAX_TOKENS = 200


@app.local_entrypoint()
def waste_demo():
    out = run_waste_remote.remote(
        prompts=WASTE_DEMO_PROMPTS, max_new_tokens=WASTE_DEMO_MAX_TOKENS
    )
    previews = out["previews"]
    eos_steps = out["eos_decode_step"]
    total = out["total_decode_steps"]

    print()
    print(f"model={MODEL_NAME}  max_new_tokens={WASTE_DEMO_MAX_TOKENS}")
    print(f"batch ran for {total} decode steps total")
    print()
    print(f"{'prompt':<52} {'needed':>7} {'wasted':>7} {'waste%':>7}")
    print("-" * 76)

    total_wasted = 0
    for prompt, eos, preview in zip(WASTE_DEMO_PROMPTS, eos_steps, previews):
        needed = eos + 1
        wasted = total - needed
        total_wasted += wasted
        short_prompt = prompt if len(prompt) <= 50 else prompt[:47] + "..."
        print(f"{short_prompt:<52} {needed:>7} {wasted:>7} {100*wasted/total:>6.1f}%")
        print(f"  → {preview!r}")

    overall_waste = total_wasted / (len(WASTE_DEMO_PROMPTS) * total)
    print()
    print(f"overall waste: {total_wasted} / {len(WASTE_DEMO_PROMPTS) * total} "
          f"steps = {100 * overall_waste:.1f}%")
    print()
    print("These are the steps that continuous batching reclaims.")
