import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import torch
from transformers import DynamicCache, StaticCache

from .model import LoadedModel


class Status(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    DONE = auto()


@dataclass
class Sequence:
    seq_id: int
    prompt: str
    status: Status = Status.WAITING
    # StaticScheduler fields
    kv_slot: int = -1
    n_prompt_tokens: int = 0
    # DynamicScheduler fields
    kv_cache: object = None
    attn_mask: torch.Tensor | None = None
    # Common
    all_ids: torch.Tensor | None = None
    text: str = ""
    eos_decode_step: int = 0
    n_decode_steps: int = 0
    submitted_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0


# ---------------------------------------------------------------------------
# Static scheduler
# ---------------------------------------------------------------------------

class StaticScheduler:
    """Pre-allocates the full KV budget (max_running x max_seq_len) at startup.

    OOM at construction means the reservation itself exceeds available VRAM —
    the wall hits before a single token is served. Finished slots are returned
    to the free list but their memory stays committed for the lifetime of the
    scheduler; a new sequence simply resets and overwrites the slot.
    """

    def __init__(
        self,
        lm: LoadedModel,
        prompts: list[str],
        max_seq_len: int = 512,
        max_running: int = 8,
        eos_token_ids: list[int] | None = None,
    ):
        self.lm = lm
        self.max_seq_len = max_seq_len
        self.max_running = max_running
        self.eos_ids = _make_eos_tensor(lm, eos_token_ids)

        # Full KV budget committed here. OOM = wall hit.
        self.kv_pool: list[StaticCache] = [
            StaticCache(
                config=lm.model.config,
                max_batch_size=1,
                max_cache_len=max_seq_len,
                device=lm.device,
                dtype=lm.dtype,
            )
            for _ in range(max_running)
        ]
        self.free_slots: deque[int] = deque(range(max_running))
        self.waiting, self.running, self.finished, self.per_step_seconds = _init_queues(prompts)

    def _admit(self) -> None:
        while self.waiting and self.free_slots:
            seq = self.waiting.popleft()
            seq.kv_slot = self.free_slots.popleft()
            seq.status = Status.PREFILLING
            seq.started_at = time.perf_counter()
            self.running.append(seq)

    @torch.inference_mode()
    def _prefill_seq(self, seq: Sequence) -> None:
        tok = self.lm.tokenizer
        enc = tok([seq.prompt], return_tensors="pt", padding=False)
        input_ids = enc.input_ids.to(self.lm.device)
        n_prompt = input_ids.shape[1]

        cache = self.kv_pool[seq.kv_slot]
        cache.reset()

        cache_position = torch.arange(n_prompt, device=self.lm.device)
        attn_mask = torch.zeros(1, self.max_seq_len, device=self.lm.device, dtype=torch.long)
        attn_mask[0, :n_prompt] = 1

        out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq.all_ids = torch.cat([input_ids, next_id], dim=1)
        seq.n_prompt_tokens = n_prompt
        seq.eos_decode_step = self.max_seq_len  # sentinel
        seq.status = Status.DONE if _is_eos(next_id, self.eos_ids) else Status.DECODING
        if seq.status == Status.DONE:
            seq.eos_decode_step = 0

    @torch.inference_mode()
    def _decode_seq(self, seq: Sequence) -> None:
        current_pos = seq.n_prompt_tokens + seq.n_decode_steps
        if current_pos >= self.max_seq_len:
            seq.eos_decode_step = seq.n_decode_steps
            seq.status = Status.DONE
            return

        cache_position = torch.tensor([current_pos], device=self.lm.device)
        attn_mask = torch.zeros(1, self.max_seq_len, device=self.lm.device, dtype=torch.long)
        attn_mask[0, :current_pos + 1] = 1

        out = self.lm.model(
            input_ids=seq.all_ids[:, -1:],
            attention_mask=attn_mask,
            past_key_values=self.kv_pool[seq.kv_slot],
            cache_position=cache_position,
            use_cache=True,
        )
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq.all_ids = torch.cat([seq.all_ids, next_id], dim=1)
        seq.n_decode_steps += 1

        if _is_eos(next_id, self.eos_ids):
            seq.eos_decode_step = seq.n_decode_steps
            seq.status = Status.DONE

    def step(self) -> None:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for seq in self.running:
            if seq.status == Status.PREFILLING:
                self._prefill_seq(seq)
        for seq in self.running:
            if seq.status == Status.DECODING:
                self._decode_seq(seq)
        torch.cuda.synchronize()
        self.per_step_seconds.append(time.perf_counter() - t0)

        now = time.perf_counter()
        done = [s for s in self.running if s.status == Status.DONE]
        for s in done:
            s.finished_at = now
            self.running.remove(s)
            self.finished.append(s)
            self.free_slots.append(s.kv_slot)  # memory stays; slot reused
        self._admit()

    def run(self) -> tuple[list[str], list[float], list[int]]:
        self._admit()
        while self.running:
            self.step()
        return _decode_results(self.lm.tokenizer, self.finished, self.per_step_seconds)


# ---------------------------------------------------------------------------
# Dynamic scheduler
# ---------------------------------------------------------------------------

class DynamicScheduler:
    """Allocates KV cache token-by-token via HuggingFace DynamicCache.

    Memory tracks actual tokens generated rather than a worst-case budget.
    No pre-allocation means no admission control: if VRAM runs out mid-run
    the process OOMs rather than rejecting the request cleanly.
    """

    def __init__(
        self,
        lm: LoadedModel,
        prompts: list[str],
        max_new_tokens: int = 512,
        max_running: int = 8,
        eos_token_ids: list[int] | None = None,
    ):
        self.lm = lm
        self.max_new_tokens = max_new_tokens
        self.max_running = max_running
        self.eos_ids = _make_eos_tensor(lm, eos_token_ids)
        self.waiting, self.running, self.finished, self.per_step_seconds = _init_queues(prompts)

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_running:
            seq = self.waiting.popleft()
            seq.status = Status.PREFILLING
            seq.started_at = time.perf_counter()
            self.running.append(seq)

    @torch.inference_mode()
    def _prefill_seq(self, seq: Sequence) -> None:
        tok = self.lm.tokenizer
        enc = tok([seq.prompt], return_tensors="pt", padding=False)
        input_ids = enc.input_ids.to(self.lm.device)
        attn_mask = enc.attention_mask.to(self.lm.device)

        out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        seq.kv_cache = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq.all_ids = torch.cat([input_ids, next_id], dim=1)
        seq.attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=self.lm.device)], dim=1)
        seq.eos_decode_step = self.max_new_tokens  # sentinel
        seq.status = Status.DONE if _is_eos(next_id, self.eos_ids) else Status.DECODING
        if seq.status == Status.DONE:
            seq.eos_decode_step = 0

    @torch.inference_mode()
    def _decode_seq(self, seq: Sequence) -> None:
        out = self.lm.model(
            input_ids=seq.all_ids[:, -1:],
            attention_mask=seq.attn_mask,
            past_key_values=seq.kv_cache,
            use_cache=True,
        )
        seq.kv_cache = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq.all_ids = torch.cat([seq.all_ids, next_id], dim=1)
        seq.attn_mask = torch.cat([seq.attn_mask, torch.ones(1, 1, device=self.lm.device)], dim=1)
        seq.n_decode_steps += 1

        if _is_eos(next_id, self.eos_ids) or seq.n_decode_steps >= self.max_new_tokens - 1:
            seq.eos_decode_step = seq.n_decode_steps
            seq.status = Status.DONE

    def step(self) -> None:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for seq in self.running:
            if seq.status == Status.PREFILLING:
                self._prefill_seq(seq)
        for seq in self.running:
            if seq.status == Status.DECODING:
                self._decode_seq(seq)
        torch.cuda.synchronize()
        self.per_step_seconds.append(time.perf_counter() - t0)

        now = time.perf_counter()
        done = [s for s in self.running if s.status == Status.DONE]
        for s in done:
            s.finished_at = now
            self.running.remove(s)
            self.finished.append(s)
        self._admit()

    def run(self) -> tuple[list[str], list[float], list[int]]:
        self._admit()
        while self.running:
            self.step()
        return _decode_results(self.lm.tokenizer, self.finished, self.per_step_seconds)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_eos_tensor(lm: LoadedModel, eos_token_ids: list[int] | None) -> torch.Tensor | None:
    if eos_token_ids is not None:
        return torch.tensor(eos_token_ids, device=lm.device)
    if lm.tokenizer.eos_token_id is not None:
        return torch.tensor([lm.tokenizer.eos_token_id], device=lm.device)
    return None


def _is_eos(token_id: torch.Tensor, eos_ids: torch.Tensor | None) -> bool:
    return eos_ids is not None and torch.isin(token_id.squeeze(), eos_ids).item()


def _init_queues(prompts: list[str]):
    now = time.perf_counter()
    waiting = deque(Sequence(seq_id=i, prompt=p, submitted_at=now) for i, p in enumerate(prompts))
    return waiting, [], [], []


def _decode_results(tokenizer, finished: list[Sequence], per_step_seconds: list[float]):
    finished.sort(key=lambda s: s.seq_id)
    for s in finished:
        s.text = tokenizer.decode(s.all_ids[0], skip_special_tokens=True)
    return (
        [s.text for s in finished],
        per_step_seconds,
        [s.eos_decode_step for s in finished],
    )
