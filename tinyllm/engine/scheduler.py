import time
from contextlib import contextmanager
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
    # TODO for Phase 3 PagedAttention: replace this with a block table
    kv_cache: object = None
    # populated by Scheduler.run()
    text: str = ""
    eos_decode_step: int = 0


@contextmanager
def _timed(log: list[float]):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    log.append(time.perf_counter() - t0)


class Scheduler:
    def __init__(
        self,
        lm: LoadedModel,
        prompts: list[str],
        max_new_tokens: int = 64,
        eos_token_ids: list[int] | None = None,
    ):
        self.lm = lm
        self.max_new_tokens = max_new_tokens
        tok = lm.tokenizer

        if eos_token_ids is not None:
            self.eos_ids: torch.Tensor | None = torch.tensor(eos_token_ids, device=lm.device)
        elif tok.eos_token_id is not None:
            self.eos_ids = torch.tensor([tok.eos_token_id], device=lm.device)
        else:
            self.eos_ids = None

        self.sequences = [Sequence(seq_id=i, prompt=p) for i, p in enumerate(prompts)]
        for s in self.sequences:
            s.eos_decode_step = max_new_tokens  # sentinel; updated when EOS fires

        self.per_step_seconds: list[float] = []

        # Batch-level tensors — allocated during _prefill().
        self._next_ids: torch.Tensor | None = None
        self._all_ids: torch.Tensor | None = None
        self._attn_mask: torch.Tensor | None = None
        self._cache = None
        self._done: torch.Tensor | None = None
        self._decode_step: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    def _record_eos(self, step: int) -> None:
        newly_done = torch.isin(self._next_ids.squeeze(-1), self.eos_ids) & ~self._done
        for i in newly_done.nonzero(as_tuple=False).squeeze(-1).tolist():
            self.sequences[i].eos_decode_step = step
        self._done |= newly_done

    def _prefill(self) -> None:
        tok = self.lm.tokenizer
        B = self.batch_size

        for s in self.sequences:
            s.status = Status.PREFILLING

        tok.padding_side = "left"
        enc = tok([s.prompt for s in self.sequences], return_tensors="pt", padding=True)
        input_ids = enc.input_ids.to(self.lm.device)
        self._attn_mask = enc.attention_mask.to(self.lm.device)
        self._done = torch.zeros(B, dtype=torch.bool, device=self.lm.device)

        with _timed(self.per_step_seconds):
            out = self.lm.model(input_ids=input_ids, attention_mask=self._attn_mask, use_cache=True)
        self._cache = out.past_key_values
        self._next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        self._all_ids = torch.cat([input_ids, self._next_ids], dim=1)
        self._attn_mask = torch.cat(
            [self._attn_mask, torch.ones(B, 1, device=self.lm.device)], dim=1
        )

        if self.eos_ids is not None:
            self._record_eos(step=0)

        for s in self.sequences:
            s.status = Status.DECODING

    # step() advances all sequences by one token / one model forward pass
    # in phase 3, we want to change this to handle sequences at different decode positions and
    # admit new sequences the moment a slot opens.
    def step(self) -> None:
        B = self.batch_size
        with _timed(self.per_step_seconds):
            out = self.lm.model(
                input_ids=self._next_ids,
                attention_mask=self._attn_mask,
                past_key_values=self._cache,
                use_cache=True,
            )
        self._cache = out.past_key_values
        self._next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        self._all_ids = torch.cat([self._all_ids, self._next_ids], dim=1)
        self._attn_mask = torch.cat(
            [self._attn_mask, torch.ones(B, 1, device=self.lm.device)], dim=1
        )
        if self.eos_ids is not None:
            self._record_eos(step=self._decode_step + 1)
        self._decode_step += 1

    # prefill, then decode all sequences until finish or max_new_tokens is reached
    def run(self) -> tuple[list[str], list[float], list[int]]:
        self._prefill()

        if self._done.all():
            texts = self._finalize(total_decode_steps=0)
            return texts, self.per_step_seconds, [s.eos_decode_step for s in self.sequences]

        for _ in range(self.max_new_tokens - 1):
            self.step()
            if self._done.all():
                break

        total_decode_steps = len(self.per_step_seconds) - 1
        texts = self._finalize(total_decode_steps)
        return texts, self.per_step_seconds, [s.eos_decode_step for s in self.sequences]

    def _finalize(self, total_decode_steps: int) -> list[str]:
        tok = self.lm.tokenizer
        texts = [tok.decode(row, skip_special_tokens=True) for row in self._all_ids]
        cap = max(total_decode_steps - 1, 0)
        for i, s in enumerate(self.sequences):
            s.text = texts[i]
            s.status = Status.DONE
            s.eos_decode_step = min(s.eos_decode_step, cap)
        return texts
