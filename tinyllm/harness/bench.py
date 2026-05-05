import statistics
from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass
class BenchResult:
    model: str
    batch_size: int
    runs: int
    tokens_per_sec: float   # for batches, this is total tok/s across all sequences in the batch
    ms_per_token: float     # for batches, this is the mean per-sequence
    ttft_ms: float
    peak_memory_mb: float


def run_benchmark(
    generate_fn: Callable[[], tuple[str, list[float]]],
    model_name: str,
    warmup: int = 1,
    runs: int = 3,
) -> BenchResult:
    def _gen():
        text, per_step = generate_fn()
        return [text], per_step

    return _run(
        generate_fn=_gen,
        model_name=model_name,
        batch_size=1,
        warmup=warmup,
        runs=runs,
    )


def run_batch_benchmark(
    generate_fn: Callable[[], tuple[list[str], list[float], list[int]]],
    model_name: str,
    batch_size: int,
    warmup: int = 1,
    runs: int = 3,
) -> BenchResult:
    def _gen():
        texts, per_step, _ = generate_fn()
        return texts, per_step

    return _run(
        generate_fn=_gen,
        model_name=model_name,
        batch_size=batch_size,
        warmup=warmup,
        runs=runs,
    )


def _run(
    generate_fn: Callable[[], tuple[list[str], list[float]]],
    model_name: str,
    batch_size: int,
    warmup: int,
    runs: int,
) -> BenchResult:
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        generate_fn()

    throughputs: list[float] = []
    ms_per_tok: list[float] = []
    ttfts: list[float] = []

    for _ in range(runs):
        _, per_step = generate_fn()
        if not per_step:
            continue
        throughputs.append(batch_size * len(per_step) / sum(per_step))
        ttfts.append(per_step[0] * 1000.0)
        if len(per_step) > 1:
            ms_per_tok.append(statistics.mean(per_step[1:]) * 1000.0)
        else:
            ms_per_tok.append(per_step[0] * 1000.0)

    return BenchResult(
        model=model_name,
        batch_size=batch_size,
        runs=runs,
        tokens_per_sec=statistics.mean(throughputs),
        ms_per_token=statistics.mean(ms_per_tok),
        ttft_ms=statistics.mean(ttfts),
        peak_memory_mb=torch.cuda.max_memory_allocated() / (1024 * 1024),
    )


def to_dict(r: BenchResult) -> dict:
    return asdict(r)


def print_result(r: BenchResult) -> None:
    print(f"model={r.model}  batch_size={r.batch_size}  runs={r.runs}")
    print(f"  tokens/sec:  {r.tokens_per_sec:.1f}  (total across batch)")
    print(f"  ms/token:    {r.ms_per_token:.2f}  (per sequence, decode)")
    print(f"  TTFT:        {r.ttft_ms:.1f} ms")
    print(f"  peak memory: {r.peak_memory_mb:.0f} MB")
