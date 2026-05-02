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
app = modal.App("tiny-decode-step6")

MODEL_NAME = "gpt2"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_remote(prompt: str, max_new_tokens: int, warmup: int, runs: int) -> dict:
    from engine.decode import generate
    from engine.model import load
    from harness.bench import run_benchmark, to_dict

    lm = load(MODEL_NAME)

    def gen() -> tuple[str, list[float]]:
        return generate(lm, prompt, max_new_tokens=max_new_tokens)

    result, text = run_benchmark(generate_fn=gen, model_name=lm.name, warmup=warmup, runs=runs)
    return {"text": text, "result": to_dict(result)}


@app.local_entrypoint()
def main(
    prompt: str = "The transformer architecture revolutionized NLP because",
    max_new_tokens: int = 64,
    warmup: int = 1,
    runs: int = 3,
    check: bool = False,
    write_snapshot: bool = False,
):
    import sys

    out = run_remote.remote(
        prompt=prompt, max_new_tokens=max_new_tokens, warmup=warmup, runs=runs
    )
    text = out["text"]
    r = out["result"]

    if check or write_snapshot:
        from harness.snapshot import check_or_write

        passed, msg = check_or_write(
            model=MODEL_NAME,
            prompt=prompt,
            n_tokens=max_new_tokens,
            text=text,
            write=write_snapshot,
        )
        print(f"\n[snapshot] {msg}")
        if not passed:
            sys.exit(1)

    print(text)
    print()
    print(f"model={r['model']} runs={r['runs']}")
    print(f"  tokens/sec:    {r['tokens_per_sec']:.2f}")
    print(f"  ms/token:      {r['ms_per_token']:.2f}  (steady-state, decode)")
    print(f"  TTFT:          {r['ttft_ms']:.1f} ms    (prefill)")
    print(f"  peak memory:   {r['peak_memory_mb']:.0f} MB")


SWEEP_CONFIGS: list[tuple[str, int]] = [
    ("The transformer architecture revolutionized NLP because", 8),
    ("The transformer architecture revolutionized NLP because", 32),
    ("The transformer architecture revolutionized NLP because", 128),
    ("The transformer architecture revolutionized NLP because", 512),
]


@app.local_entrypoint()
def sweep(warmup: int = 1, runs: int = 3):
    rows: list[tuple[str, int, dict]] = []
    for prompt, n in SWEEP_CONFIGS:
        out = run_remote.remote(prompt=prompt, max_new_tokens=n, warmup=warmup, runs=runs)
        rows.append((prompt, n, out["result"]))

    print()
    print("| prompt | new_tokens | tok/s | ms/tok | TTFT (ms) | peak MB |")
    print("|---|---:|---:|---:|---:|---:|")
    for prompt, n, r in rows:
        short = prompt if len(prompt) <= 40 else prompt[:37] + "..."
        print(
            f"| {short} | {n} | {r['tokens_per_sec']:.1f} | "
            f"{r['ms_per_token']:.2f} | {r['ttft_ms']:.0f} | "
            f"{r['peak_memory_mb']:.0f} |"
        )
