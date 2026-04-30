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
    .add_local_python_source("generate")
)

hf_cache = modal.Volume.from_name("tiny-decode-hf-cache", create_if_missing=True)
app = modal.App("tiny-decode-step4")


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=600)
def run_remote(prompt: str, max_new_tokens: int, warmup: int, runs: int) -> dict:
    import statistics

    from generate import load_model, generate

    lm = load_model()

    for _ in range(warmup):
        generate(lm, prompt, max_new_tokens=max_new_tokens)

    throughputs: list[float] = []
    ttfts_ms: list[float] = []
    steady_ms: list[float] = []
    text = ""
    for _ in range(runs):
        text, per_token_seconds = generate(lm, prompt, max_new_tokens=max_new_tokens)
        throughputs.append(len(per_token_seconds) / sum(per_token_seconds))
        ttfts_ms.append(per_token_seconds[0] * 1000.0)
        if len(per_token_seconds) > 1:
            steady_ms.append(statistics.mean(per_token_seconds[1:]) * 1000.0)

    return {
        "text": text,
        "tokens_per_sec": statistics.mean(throughputs),
        "ttft_ms": statistics.mean(ttfts_ms),
        "ms_per_token": statistics.mean(steady_ms) if steady_ms else float("nan"),
    }


@app.local_entrypoint()
def main(
    prompt: str = "The transformer architecture revolutionized NLP because",
    max_new_tokens: int = 64,
    warmup: int = 1,
    runs: int = 3,
):
    out = run_remote.remote(
        prompt=prompt, max_new_tokens=max_new_tokens, warmup=warmup, runs=runs
    )
    print(out["text"])
    print(f"\ntokens/sec:  {out['tokens_per_sec']:.2f}")
    print(f"TTFT:        {out['ttft_ms']:.1f} ms   (prefill)")
    print(f"steady ms/t: {out['ms_per_token']:.2f} ms  (decode)")
