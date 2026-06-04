"""Abstract pruner interface for evalscope benchmarks."""
from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalscope.api.dataset import Sample


class BasePruner(abc.ABC):
    """Reduce a list of Samples to a representative subset.

    All subclasses must implement :meth:`prune`.  The contract is simple:
    receive the full list, return a shorter list.  Callers wrap the result
    in a ``MemoryDataset`` to plug back into the evalscope pipeline.
    """

    @abc.abstractmethod
    def prune(
        self,
        samples: list[Any],
        prune_ratio: float,
        *,
        seed: int = 42,
    ) -> list[Any]:
        """Return a subset of *samples* of length ``ceil(N * (1 - prune_ratio))``.

        Args:
            samples:     Full list of ``Sample`` objects (or plain dicts for tests).
            prune_ratio: Fraction to *remove*, in ``(0, 1)``.
            seed:        RNG seed for reproducibility.
        """

    @staticmethod
    def target_k(n: int, prune_ratio: float) -> int:
        return max(1, math.ceil(n * (1.0 - prune_ratio)))
