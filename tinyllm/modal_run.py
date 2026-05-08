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
app = modal.App("tiny-decode-phase4-step3")

DRAFT_MODEL = "Qwen/Qwen3-0.6B"
TARGET_MODEL = "Qwen/Qwen3-4B"
SPEC_K = 4
NEW_TOKENS = 128
TEMPERATURE_SWEEP = [0.0, 0.3, 0.7, 1.0, 1.5]

EASY_PROMPTS = [
    "What is the capital of France?",
    "Is Python interpreted or compiled? Answer in one sentence.",
    "What is 2 + 2?",
    "Name the three primary colors.",
]
HARD_PROMPTS = [
    "Write a short story about a robot who learns to paint.",
    "Describe the history of the Roman Empire in significant detail.",
    "Compose a poem about the feeling of autumn rain.",
    "Explain the philosophical implications of the trolley problem.",
]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_acceptance_remote(
    prompts: list[str],
    new_tokens: int,
    K: int,
    temperature: float,
    runs: int,
) -> list[dict]:
    import statistics

    from engine.model import format_prompt, load
    from spec_dec.generate import spec_generate

    target_lm = load(TARGET_MODEL)
    draft_lm = load(DRAFT_MODEL)
    rows = []
    for prompt in prompts:
        formatted = format_prompt(target_lm.tokenizer, prompt)
        acceptance_rates = []
        tok_per_sec = []
        for _ in range(runs):
            _, per_round, acc = spec_generate(
                target_lm, draft_lm, formatted, new_tokens, K, temperature
            )
            acceptance_rates.append(acc)
            if per_round:
                n_est = len(per_round) * (K * acc + 1)
                tok_per_sec.append(n_est / sum(per_round))
        rows.append(
            {
                "prompt": prompt,
                "acceptance_rate": statistics.mean(acceptance_rates),
                "tokens_per_sec": statistics.mean(tok_per_sec) if tok_per_sec else 0.0,
            }
        )
    return rows


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
)
def run_temperature_remote(
    prompt: str,
    temperatures: list[float],
    new_tokens: int,
    K: int,
    runs: int,
) -> list[dict]:
    import statistics

    from engine.model import format_prompt, load
    from spec_dec.generate import spec_generate

    target_lm = load(TARGET_MODEL)
    draft_lm = load(DRAFT_MODEL)
    formatted = format_prompt(target_lm.tokenizer, prompt)
    rows = []
    for T in temperatures:
        acceptance_rates = []
        for _ in range(runs):
            _, _, acc = spec_generate(target_lm, draft_lm, formatted, new_tokens, K, T)
            acceptance_rates.append(acc)
        rows.append({"temperature": T, "acceptance_rate": statistics.mean(acceptance_rates)})
    return rows


@app.local_entrypoint()
def acceptance_demo(
    k: int = SPEC_K,
    temperature: float = 1.0,
    new_tokens: int = NEW_TOKENS,
    runs: int = 3,
):
    easy_future = run_acceptance_remote.spawn(EASY_PROMPTS, new_tokens, k, temperature, runs)
    hard_future = run_acceptance_remote.spawn(HARD_PROMPTS, new_tokens, k, temperature, runs)
    easy_rows = easy_future.get()
    hard_rows = hard_future.get()

    print(
        f"\nmodel={TARGET_MODEL}  draft={DRAFT_MODEL}  "
        f"K={k}  T={temperature}  new_tokens={new_tokens}\n"
    )

    for label, rows in [("EASY", easy_rows), ("HARD", hard_rows)]:
        import statistics
        print(f"{label} PROMPTS")
        print(f"  {'prompt':<48}  {'accept':>7}  {'tok/s':>7}")
        print("  " + "-" * 66)
        for r in rows:
            short = r["prompt"] if len(r["prompt"]) <= 46 else r["prompt"][:43] + "..."
            print(
                f"  {short:<48}  {r['acceptance_rate']:>6.1%}  {r['tokens_per_sec']:>7.1f}"
            )
        avg_acc = statistics.mean(r["acceptance_rate"] for r in rows)
        avg_tps = statistics.mean(r["tokens_per_sec"] for r in rows)
        print(f"  {'AVERAGE':<48}  {avg_acc:>6.1%}  {avg_tps:>7.1f}\n")


@app.local_entrypoint()
def temperature_demo(
    k: int = SPEC_K,
    new_tokens: int = NEW_TOKENS,
    runs: int = 3,
):
    prompt = "Explain how neural networks learn from data."
    rows = run_temperature_remote.remote(prompt, TEMPERATURE_SWEEP, new_tokens, k, runs)

    print(f"\nmodel={TARGET_MODEL}  draft={DRAFT_MODEL}  K={k}")
    print(f"prompt: {prompt!r}\n")
    print(f"  {'temperature':>12}  {'acceptance':>11}")
    print("  " + "-" * 26)
    for r in rows:
        bar = "█" * int(r["acceptance_rate"] * 20)
        print(f"  {r['temperature']:>12.1f}  {r['acceptance_rate']:>10.1%}  {bar}")
