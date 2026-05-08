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
    .add_local_python_source("engine", "harness", "spec_dec")
)

hf_cache = modal.Volume.from_name("tiny-decode-hf-cache", create_if_missing=True)
app = modal.App("tiny-decode-phase4-step2")

DRAFT_MODEL = "Qwen/Qwen3-0.6B"
TARGET_MODEL = "Qwen/Qwen3-4B"
SPEC_K = 4

SNAPSHOT_PROMPTS = [
    "The transformer architecture revolutionized NLP because",
    "Explain the difference between supervised and unsupervised learning.",
    "Once upon a time, a curious fox discovered",
]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_snapshot_remote(prompt: str, new_tokens: int) -> dict:
    from engine.decode import generate
    from engine.model import format_prompt, load
    from spec_dec.generate import spec_generate

    target_lm = load(TARGET_MODEL)
    draft_lm = load(DRAFT_MODEL)
    formatted = format_prompt(target_lm.tokenizer, prompt)

    target_text, _ = generate(target_lm, formatted, new_tokens)
    spec_text, _, acceptance_rate = spec_generate(
        target_lm, draft_lm, formatted, new_tokens, K=SPEC_K
    )

    return {
        "target_text": target_text,
        "spec_text": spec_text,
        "match": target_text == spec_text,
        "acceptance_rate": acceptance_rate,
    }


@app.local_entrypoint()
def snapshot(new_tokens: int = 64):
    print(f"\nSnapshot check — new_tokens={new_tokens}  K={SPEC_K}\n")

    all_pass = True
    for prompt in SNAPSHOT_PROMPTS:
        out = run_snapshot_remote.remote(prompt, new_tokens)
        status = "PASS" if out["match"] else "FAIL"
        if not out["match"]:
            all_pass = False
        print(
            f"  [{status}]  acceptance={out['acceptance_rate']:.1%}  "
            f"prompt={prompt[:50]!r}"
        )
        if not out["match"]:
            print(f"         target: {out['target_text'][:120]!r}")
            print(f"         spec:   {out['spec_text'][:120]!r}")

    print()
    if all_pass:
        print("All snapshots match. Spec dec produces identical output to target-only decoding.")
    else:
        print("SNAPSHOT MISMATCH — check spec_dec/sampler.py acceptance logic.")


