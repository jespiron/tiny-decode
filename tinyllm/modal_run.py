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
app = modal.App("tiny-decode-step1")


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache}, timeout=600)
def run_remote(prompt: str, max_new_tokens: int) -> str:
    from generate import load_model, generate

    lm = load_model()
    return generate(lm, prompt, max_new_tokens=max_new_tokens)


@app.local_entrypoint()
def main(
    prompt: str = "The transformer architecture revolutionized NLP because",
    max_new_tokens: int = 64,
):
    print(run_remote.remote(prompt=prompt, max_new_tokens=max_new_tokens))
