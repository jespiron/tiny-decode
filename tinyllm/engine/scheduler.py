import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import torch

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
    kv_cache: object = None            # DynamicCache owned by this sequence
    all_ids: torch.Tensor | None = None
    attn_mask: torch.Tensor | None = None
    text: str = ""
    eos_decode_step: int = 0           # 0 = EOS in prefill output, k = kth decode step
    n_decode_steps: int = 0
    submitted_at: float = 0.0          # wall time when request entered queue
    started_at: float = 0.0            # wall time when prefill began
    finished_at: float = 0.0           # wall time when generation completed


class Scheduler:
    def __init__(
        self,
        lm: LoadedModel,
        prompts: list[str],
        max_new_tokens: int = 64,
        max_running: int = 8,
        eos_token_ids: list[int] | None = None,
    ):
        self.lm = lm
        self.max_new_tokens = max_new_tokens
        self.max_running = max_running
        tok = lm.tokenizer

        if eos_token_ids is not None:
            self.eos_ids: torch.Tensor | None = torch.tensor(eos_token_ids, device=lm.device)
        elif tok.eos_token_id is not None:
            self.eos_ids = torch.tensor([tok.eos_token_id], device=lm.device)
        else:
            self.eos_ids = None

        now = time.perf_counter()
        self.waiting: deque[Sequence] = deque(
            Sequence(seq_id=i, prompt=p, submitted_at=now)
            for i, p in enumerate(prompts)
        )
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.per_step_seconds: list[float] = []

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_running:
            seq = self.waiting.popleft()
            seq.status = Status.PREFILLING
            seq.started_at = time.perf_counter()
            self.running.append(seq)

    def _prefill_seq(self, seq: Sequence) -> None:
        tok = self.lm.tokenizer
        enc = tok([seq.prompt], return_tensors="pt", padding=False)
        input_ids = enc.input_ids.to(self.lm.device)
        attn_mask = enc.attention_mask.to(self.lm.device)

        out = self.lm.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        seq.kv_cache = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        seq.all_ids = torch.cat([input_ids, next_id], dim=1)
        seq.attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=self.lm.device)], dim=1)

        is_eos = (
            self.eos_ids is not None
            and torch.isin(next_id.squeeze(), self.eos_ids).item()
        )
        if is_eos:
            seq.eos_decode_step = 0
            seq.status = Status.DONE
        else:
            seq.eos_decode_step = self.max_new_tokens  # sentinel
            seq.status = Status.DECODING

    def _decode_seq(self, seq: Sequence) -> None:
        last_id = seq.all_ids[:, -1:]  # [1, 1]
        out = self.lm.model(
            input_ids=last_id,
            attention_mask=seq.attn_mask,
            past_key_values=seq.kv_cache,
            use_cache=True,
        )
        seq.kv_cache = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq.all_ids = torch.cat([seq.all_ids, next_id], dim=1)
        seq.attn_mask = torch.cat([seq.attn_mask, torch.ones(1, 1, device=self.lm.device)], dim=1)
        seq.n_decode_steps += 1

        is_eos = (
            self.eos_ids is not None
            and torch.isin(next_id.squeeze(), self.eos_ids).item()
        )
        if is_eos or seq.n_decode_steps >= self.max_new_tokens - 1:
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

        # retire finished sequences and immediately fill their slots
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

        tok = self.lm.tokenizer
        self.finished.sort(key=lambda s: s.seq_id)
        for s in self.finished:
            s.text = tok.decode(s.all_ids[0], skip_special_tokens=True)

        texts = [s.text for s in self.finished]
        eos_decode_step = [s.eos_decode_step for s in self.finished]
        return texts, self.per_step_seconds, eos_decode_step
