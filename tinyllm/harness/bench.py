import statistics
from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass
class BenchResult:
    model: str
    runs: int
    tokens_per_sec: float
    ms_per_token: float
    ttft_ms: float
    peak_memory_mb: float


def run_benchmark(
    generate_fn: Callable[[], tuple[str, list[float]]],
    model_name: str,
    warmup: int = 1,
    runs: int = 3,
) -> tuple[BenchResult, str]:
    """Run generate_fn (warmup + runs) times. Returns (result, last_generated_text)."""
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        generate_fn()

    throughputs: list[float] = []
    ttfts_ms: list[float] = []
    steady_ms: list[float] = []
    text = ""
    for _ in range(runs):
        text, per_token_seconds = generate_fn()
        throughputs.append(len(per_token_seconds) / sum(per_token_seconds))
        ttfts_ms.append(per_token_seconds[0] * 1000.0)
        if len(per_token_seconds) > 1:
            steady_ms.append(statistics.mean(per_token_seconds[1:]) * 1000.0)

    result = BenchResult(
        model=model_name,
        runs=runs,
        tokens_per_sec=statistics.mean(throughputs),
        ms_per_token=statistics.mean(steady_ms) if steady_ms else statistics.mean(ttfts_ms),
        ttft_ms=statistics.mean(ttfts_ms),
        peak_memory_mb=torch.cuda.max_memory_allocated() / (1024 * 1024),
    )
    return result, text


def to_dict(r: BenchResult) -> dict:
    return asdict(r)


def print_result(r: BenchResult) -> None:
    print(f"model={r.model} runs={r.runs}")
    print(f"  tokens/sec:    {r.tokens_per_sec:.2f}")
    print(f"  ms/token:      {r.ms_per_token:.2f}  (steady-state, decode)")
    print(f"  TTFT:          {r.ttft_ms:.1f} ms    (prefill)")
    print(f"  peak memory:   {r.peak_memory_mb:.0f} MB")
