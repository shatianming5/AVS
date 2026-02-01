from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    patch_size: int = 16

    def tokens_for_resolution(self, resolution: int) -> int:
        if resolution % self.patch_size != 0:
            raise ValueError(f"resolution {resolution} must be divisible by patch_size {self.patch_size}")
        grid = resolution // self.patch_size
        return int(grid * grid)

