from collections import deque
from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .kv_cache import BlockAllocator


# ---------------------------------------------------------------------------
# BlockAllocator (identical to step 2)
# ---------------------------------------------------------------------------

class BlockAllocator:
    def __init__(
        self,
        n_blocks: int,
        n_layers: int,
        n_kv_heads: int,
        block_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.block_size = block_size
        self.n_layers = n_layers
        self.kv_store = torch.zeros(
            n_blocks, n_layers, 2, n_kv_heads, block_size, head_dim,
            device=device, dtype=dtype,
        )
        self._free_list: deque[int] = deque(range(n_blocks))
        self._n_allocated = 0
        self._peak_allocated = 0

    def alloc(self) -> int:
        if not self._free_list:
            raise RuntimeError("BlockAllocator: out of free blocks")
        block_id = self._free_list.popleft()
        self._n_allocated += 1
        if self._n_allocated > self._peak_allocated:
            self._peak_allocated = self._n_allocated
        return block_id

    def free(self, block_id: int) -> None:
        self._free_list.append(block_id)
        self._n_allocated -= 1

    @property
    def n_blocks(self) -> int:
        return self.kv_store.shape[0]

    @property
    def n_free(self) -> int:
        return len(self._free_list)

    @property
    def n_allocated(self) -> int:
        return self._n_allocated

    @property
    def peak_allocated(self) -> int:
        return self._peak_allocated

    @classmethod
    def from_model(cls, model, block_size, n_blocks, device, dtype) -> "BlockAllocator":
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        return cls(
            n_blocks=n_blocks,
            n_layers=cfg.num_hidden_layers,
            n_kv_heads=cfg.num_key_value_heads,
            block_size=block_size,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def budget_n_blocks(model, block_size: int, vram_fraction: float = 0.8) -> int:
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        bytes_per_block = (
            cfg.num_hidden_layers * 2 * cfg.num_key_value_heads
            * block_size * head_dim * 2
        )
        available = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()
        ) * vram_fraction
        return max(1, int(available / bytes_per_block))


# ---------------------------------------------------------------------------
# BlockTable (new in step 3)
# ---------------------------------------------------------------------------

@dataclass
class BlockTable:
    block_size: int
    block_ids: list[int] = field(default_factory=list)
    n_filled: int = 0  # total token slots written so far

    def maybe_extend(self, allocator: "BlockAllocator") -> None:
        if self.n_filled % self.block_size == 0:
            self.block_ids.append(allocator.alloc())

    def record_token(self) -> None:
        self.n_filled += 1

    def physical_pos(self, token_pos: int) -> tuple[int, int]:
        logical_block = token_pos // self.block_size
        offset = token_pos % self.block_size
        return self.block_ids[logical_block], offset

    def free_all(self, allocator: "BlockAllocator") -> None:
        for block_id in self.block_ids:
            allocator.free(block_id)
        self.block_ids.clear()
        self.n_filled = 0

    def describe(self, block_size: int | None = None) -> list[dict]:
        bs = block_size or self.block_size
        rows = []
        for logical_idx, phys_id in enumerate(self.block_ids):
            start = logical_idx * bs
            end_max = (logical_idx + 1) * bs
            filled = min(self.n_filled - start, bs)
            rows.append({
                "logical": logical_idx,
                "physical": phys_id,
                "token_range": f"{start}–{min(end_max, self.n_filled) - 1}",
                "filled": filled,
                "capacity": bs,
            })
        return rows
