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
app = modal.App("tiny-decode-step3")


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=600)
def run_remote(prompt: str, max_new_tokens: int, warmup: int, runs: int) -> dict:
    import statistics

    from generate import load_model, generate

    lm = load_model()

    warmup_throughputs: list[float] = []
    for _ in range(warmup):
        _, per_token_seconds = generate(lm, prompt, max_new_tokens=max_new_tokens)
        warmup_throughputs.append(len(per_token_seconds) / sum(per_token_seconds))

    throughputs: list[float] = []
    text = ""
    for _ in range(runs):
        text, per_token_seconds = generate(lm, prompt, max_new_tokens=max_new_tokens)
        throughputs.append(len(per_token_seconds) / sum(per_token_seconds))

    return {
        "text": text,
        "tokens_per_sec_warmup": warmup_throughputs,
        "tokens_per_sec_mean": statistics.mean(throughputs),
        "tokens_per_sec_per_run": throughputs,
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
    warmup_str = ", ".join(f"{t:.2f}" for t in out["tokens_per_sec_warmup"])
    per_run = ", ".join(f"{t:.2f}" for t in out["tokens_per_sec_per_run"])
    print(f"\nwarmup (discarded): [{warmup_str}] tok/s")
    print(f"per-run tok/s:      [{per_run}]")
    print(f"mean:               {out['tokens_per_sec_mean']:.2f} tok/s")
