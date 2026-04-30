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
app = modal.App("tiny-decode-step2")


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=600)
def run_remote(prompt: str, max_new_tokens: int) -> dict:
    from generate import load_model, generate

    lm = load_model()
    text, per_token_seconds = generate(lm, prompt, max_new_tokens=max_new_tokens)
    return {
        "text": text,
        "tokens_per_sec": len(per_token_seconds) / sum(per_token_seconds),
    }


@app.local_entrypoint()
def main(
    prompt: str = "The transformer architecture revolutionized NLP because",
    max_new_tokens: int = 64,
):
    out = run_remote.remote(prompt=prompt, max_new_tokens=max_new_tokens)
    print(out["text"])
    print(f"\n{out['tokens_per_sec']:.2f} tok/s")
