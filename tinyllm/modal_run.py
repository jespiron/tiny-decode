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
app = modal.App("tiny-decode-phase4-step1")

DRAFT_MODEL = "Qwen/Qwen3-0.6B"
TARGET_MODEL = "Qwen/Qwen3-4B"
NEW_TOKENS = 128
WARMUP_PROMPT = "The transformer architecture revolutionized NLP because"

PROMPTS = [
    "Explain how neural networks learn from data.",
    "Once upon a time in a kingdom by the sea,",
    "The key insight behind the attention mechanism is",
    "Gradient descent works by iteratively adjusting",
]


def _profile_model(lm, new_tokens: int) -> dict:
    """Runs one warmup prompt (discarded) then PROMPTS. Both models run in the
    same container so the ratio is measured on the same physical bus.
    """
    import statistics

    import torch
    from engine.decode import generate
    from engine.model import format_prompt

    # Warmup: one pass to bring GPU to steady state before measuring.
    generate(lm, format_prompt(lm.tokenizer, WARMUP_PROMPT), new_tokens)
    torch.cuda.reset_peak_memory_stats()

    all_per_tok_ms: list[list[float]] = []
    for prompt in PROMPTS:
        formatted = format_prompt(lm.tokenizer, prompt)
        _, per_tok = generate(lm, formatted, new_tokens)
        all_per_tok_ms.append([t * 1000.0 for t in per_tok])

    decode_times = [t for per_tok in all_per_tok_ms for t in per_tok[1:]]

    return {
        "model": lm.name,
        "new_tokens": new_tokens,
        "n_prompts": len(PROMPTS),
        "ttft_ms": statistics.mean(per_tok[0] for per_tok in all_per_tok_ms),
        "decode_ms_mean": statistics.mean(decode_times),
        "decode_ms_p50": sorted(decode_times)[len(decode_times) // 2],
        "decode_ms_p90": sorted(decode_times)[int(len(decode_times) * 0.9)],
        "tokens_per_sec": 1000.0 / statistics.mean(decode_times),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "per_prompt_per_tok_ms": all_per_tok_ms,
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_profile_remote(new_tokens: int) -> tuple[dict, dict]:
    """Profile draft and target on the same container — same physical bus."""
    from engine.model import load

    draft_lm = load(DRAFT_MODEL)
    draft_out = _profile_model(draft_lm, new_tokens)
    del draft_lm

    target_lm = load(TARGET_MODEL)
    target_out = _profile_model(target_lm, new_tokens)
    del target_lm

    return draft_out, target_out


@app.local_entrypoint()
def profile(new_tokens: int = NEW_TOKENS):
    """Summary table: draft vs. target latency and throughput.

    Both models run sequentially in the same container so the ratio is
    measured on the same physical memory bus — no cross-container noise.
    """
    print(f"\nRunning latency profile on A10G  (new_tokens={new_tokens})\n")

    draft_out, target_out = run_profile_remote.remote(new_tokens)

    header = (
        f"{'model':<25}  {'tok/s':>7}  {'ms/tok (mean)':>14}"
        f"  {'ms/tok (p50)':>12}  {'ms/tok (p90)':>12}  {'TTFT (ms)':>10}"
    )
    print(header)
    print("-" * len(header))
    for out in [draft_out, target_out]:
        print(
            f"{out['model']:<25}  "
            f"{out['tokens_per_sec']:>7.1f}  "
            f"{out['decode_ms_mean']:>14.2f}  "
            f"{out['decode_ms_p50']:>12.2f}  "
            f"{out['decode_ms_p90']:>12.2f}  "
            f"{out['ttft_ms']:>10.1f}"
        )

