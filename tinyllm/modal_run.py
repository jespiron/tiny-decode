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
app = modal.App("tiny-decode-phase4-step4")

DRAFT_MODEL = "Qwen/Qwen3-0.6B"
TARGET_MODEL = "Qwen/Qwen3-4B"
K_VALUES = [1, 2, 4, 8, 16]
TEMPERATURE = 1.0
NEW_TOKENS = 128

EASY_PROMPT = (
    "What is the capital of France? Please also name three other major French "
    "cities and describe what each is known for."
)
HARD_PROMPT = (
    "Write a short story about a robot who learns to paint. "
    "Make it emotionally resonant and surprising."
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_k_sweep_remote(
    prompt: str,
    k_values: list[int],
    new_tokens: int,
    temperature: float,
    warmup: int,
    runs: int,
) -> list[dict]:
    import statistics

    from engine.decode import generate
    from engine.model import format_prompt, load
    from spec_dec.generate import spec_generate

    target_lm = load(TARGET_MODEL)
    draft_lm = load(DRAFT_MODEL)
    formatted = format_prompt(target_lm.tokenizer, prompt)
    results = []

    # K=0 baseline: target-only
    baseline_tps = []
    for i in range(warmup + runs):
        _, per_tok = generate(target_lm, formatted, new_tokens)
        if per_tok and i >= warmup:
            baseline_tps.append(len(per_tok) / sum(per_tok))
    results.append(
        {
            "K": 0,
            "tokens_per_sec": statistics.mean(baseline_tps),
            "ms_per_token": 1000.0 / statistics.mean(baseline_tps),
            "acceptance_rate": None,
        }
    )

    for K in k_values:
        tps_list, acc_list = [], []
        for i in range(warmup + runs):
            _, per_round, acc = spec_generate(
                target_lm, draft_lm, formatted, new_tokens, K, temperature
            )
            if not per_round or i < warmup:
                continue
            n_est = len(per_round) * (K * acc + 1)
            tps_list.append(n_est / sum(per_round))
            acc_list.append(acc)

        results.append(
            {
                "K": K,
                "tokens_per_sec": statistics.mean(tps_list) if tps_list else 0.0,
                "ms_per_token": 1000.0 / statistics.mean(tps_list) if tps_list else 0.0,
                "acceptance_rate": statistics.mean(acc_list) if acc_list else 0.0,
            }
        )

    return results


@app.local_entrypoint()
def sweep(
    new_tokens: int = NEW_TOKENS,
    temperature: float = TEMPERATURE,
    warmup: int = 1,
    runs: int = 3,
):
    easy_future = run_k_sweep_remote.spawn(
        EASY_PROMPT, K_VALUES, new_tokens, temperature, warmup, runs
    )
    hard_future = run_k_sweep_remote.spawn(
        HARD_PROMPT, K_VALUES, new_tokens, temperature, warmup, runs
    )
    easy_rows = easy_future.get()
    hard_rows = hard_future.get()

    print(
        f"\nmodel={TARGET_MODEL}  draft={DRAFT_MODEL}  "
        f"T={temperature}  new_tokens={new_tokens}\n"
    )

    header = (
        f"{'K':>4}  "
        f"{'easy tok/s':>10}  {'easy ms/tok':>11}  {'easy accept':>11}  "
        f"{'hard tok/s':>10}  {'hard ms/tok':>11}  {'hard accept':>11}"
    )
    print(header)
    print("-" * len(header))

    for easy, hard in zip(easy_rows, hard_rows):
        K_label = str(easy["K"]) if easy["K"] > 0 else "0 (baseline)"
        easy_acc = f"{easy['acceptance_rate']:.1%}" if easy["acceptance_rate"] is not None else "—"
        hard_acc = f"{hard['acceptance_rate']:.1%}" if hard["acceptance_rate"] is not None else "—"
        print(
            f"{K_label:>4}  "
            f"{easy['tokens_per_sec']:>10.1f}  {easy['ms_per_token']:>11.2f}  {easy_acc:>11}  "
            f"{hard['tokens_per_sec']:>10.1f}  {hard['ms_per_token']:>11.2f}  {hard_acc:>11}"
        )

    print()
    baseline_ms = easy_rows[0]["ms_per_token"]
    best_easy = max(easy_rows[1:], key=lambda r: r["tokens_per_sec"])
    best_hard = max(hard_rows[1:], key=lambda r: r["tokens_per_sec"])
    print(
        f"Best K for easy text: K={best_easy['K']}  "
        f"({baseline_ms / best_easy['ms_per_token']:.2f}× vs baseline)"
    )
    print(
        f"Best K for hard text: K={best_hard['K']}  "
        f"({baseline_ms / best_hard['ms_per_token']:.2f}× vs baseline)"
    )


