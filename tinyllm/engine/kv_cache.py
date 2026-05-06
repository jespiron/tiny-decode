from collections import deque
from math import ceil

import torch


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

    # ------------------------------------------------------------------
    # Pool operations

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

    # ------------------------------------------------------------------
    # Introspection

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

    # ------------------------------------------------------------------
    # Construction helper

    @classmethod
    def from_model(
        cls,
        model,
        block_size: int,
        n_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "BlockAllocator":
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
        import torch
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        bytes_per_block = (
            cfg.num_hidden_layers * 2 * cfg.num_key_value_heads
            * block_size * head_dim * 2  # fp16
        )
        available = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()
        ) * vram_fraction
        return max(1, int(available / bytes_per_block))
