from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# BlockAllocator (identical to steps 2 and 3)
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
# BlockTable (identical to step 3)
# ---------------------------------------------------------------------------

@dataclass
class BlockTable:
    block_size: int
    block_ids: list[int] = field(default_factory=list)
    n_filled: int = 0

    def maybe_extend(self, allocator: BlockAllocator) -> None:
        if self.n_filled % self.block_size == 0:
            self.block_ids.append(allocator.alloc())

    def record_token(self) -> None:
        self.n_filled += 1

    def physical_pos(self, token_pos: int) -> tuple[int, int]:
        return self.block_ids[token_pos // self.block_size], token_pos % self.block_size

    def free_all(self, allocator: BlockAllocator) -> None:
        for block_id in self.block_ids:
            allocator.free(block_id)
        self.block_ids.clear()
        self.n_filled = 0


# ---------------------------------------------------------------------------
# PagedCache (new in step 4)
# ---------------------------------------------------------------------------

class PagedCache:
    def __init__(
        self,
        block_table: BlockTable,
        allocator: BlockAllocator,
        n_layers: int,
    ):
        self.block_table = block_table
        self.allocator = allocator
        self.n_layers = n_layers
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = key_states.shape[2]
        start_pos = self._seen_tokens

        if layer_idx == 0:
            for i in range(seq_len):
                pos = start_pos + i
                if pos % self.allocator.block_size == 0:
                    self.block_table.block_ids.append(self.allocator.alloc())

        # Write each new token's K/V into its block slot.
        for i in range(seq_len):
            pos = start_pos + i
            block_idx = pos // self.allocator.block_size
            offset = pos % self.allocator.block_size
            phys = self.block_table.block_ids[block_idx]
            self.allocator.kv_store[phys, layer_idx, 0, :, offset, :] = key_states[0, :, i, :]
            self.allocator.kv_store[phys, layer_idx, 1, :, offset, :] = value_states[0, :, i, :]

        if layer_idx == self.n_layers - 1:
            self._seen_tokens += seq_len
            self.block_table.n_filled = self._seen_tokens

        n_filled = start_pos + seq_len
        return self._gather(layer_idx, n_filled)

    def _gather(
        self, layer_idx: int, n_filled: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = self.allocator.block_size
        n_full = n_filled // bs
        remainder = n_filled % bs

        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []

        for i, phys in enumerate(self.block_table.block_ids):
            if i < n_full:
                k_chunks.append(self.allocator.kv_store[phys, layer_idx, 0])
                v_chunks.append(self.allocator.kv_store[phys, layer_idx, 1])
            elif i == n_full and remainder > 0:
                k_chunks.append(self.allocator.kv_store[phys, layer_idx, 0, :, :remainder, :])
                v_chunks.append(self.allocator.kv_store[phys, layer_idx, 1, :, :remainder, :])
                break

        k_out = torch.cat(k_chunks, dim=1).unsqueeze(0)
        v_out = torch.cat(v_chunks, dim=1).unsqueeze(0)
        return k_out, v_out

    # ------------------------------------------------------------------
    # HuggingFace Cache interface

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_mask_sizes(self, q_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self._seen_tokens + q_length, 0
