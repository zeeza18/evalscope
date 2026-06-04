"""Abstract interface for benchmark pruners."""
from __future__ import annotations

import abc
from typing import Any


class BasePruner(abc.ABC):
    """
    Minimal interface that every pruner must implement.

    evalscope dataset adapters call `prune(samples, prune_ratio, **kwargs)`
    and receive back the selected subset.  Nothing else is required.
    """

    @abc.abstractmethod
    def prune(
        self,
        samples: list[dict[str, Any]],
        prune_ratio: float,
        *,
        seed: int = 42,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Return a subset of `samples` of length ceil(len(samples) * (1 - prune_ratio)).

        Args:
            samples:     Full list of dataset records (dicts with at least an `index` field).
            prune_ratio: Fraction to remove, in (0, 1).  0.9 → keep 10 %.
            seed:        RNG seed for reproducibility.
        """

    # ------------------------------------------------------------------
    # Helpers shared by all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _target_k(n: int, prune_ratio: float) -> int:
        import math
        keep = max(1, math.ceil(n * (1.0 - prune_ratio)))
        return keep
