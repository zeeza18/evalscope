"""
Universal base adapter for pruned benchmarks.

Any evalscope dataset adapter can inherit from this class, implement
`_load_full_samples()`, and automatically get:
  - stratified-diversity pruning driven by --dataset-args
  - reproducible subset via a fixed seed
  - transparent logging of what was pruned
"""
from __future__ import annotations

import logging
from typing import Any

from evalscope_ext.pruners.stratified import StratifiedDiversityPruner

logger = logging.getLogger(__name__)

_STRATEGY_MAP = {
    "stratified_diversity": StratifiedDiversityPruner,
}


class PrunedDatasetAdapter:
    """
    Mixin / base class for pruned benchmark adapters.

    Subclasses must set:
        BENCHMARK_NAME : str   — key into the feature-extractor registry
        DATASET_ID     : str   — name used in evalscope CLI
        SCORE_FIELD    : str   — field name for the per-sample score
    """

    BENCHMARK_NAME: str = "generic"
    DATASET_ID: str = "pruned"
    SCORE_FIELD: str = "score"

    def __init__(self, model_args: Any = None, dataset_args: dict | None = None):
        self.model_args = model_args
        self.dataset_args: dict[str, Any] = dataset_args or {}

    # ------------------------------------------------------------------
    # Public API (called by evalscope internals)
    # ------------------------------------------------------------------

    def load_dataset(self) -> list[dict[str, Any]]:
        samples = self._load_full_samples()
        prune_ratio: float = float(self.dataset_args.get("prune_ratio", 0.5))
        strategy_name: str = self.dataset_args.get("pruning_strategy", "stratified_diversity")
        seed: int = int(self.dataset_args.get("seed", 42))

        strategy_cls = _STRATEGY_MAP.get(strategy_name, StratifiedDiversityPruner)
        pruner = strategy_cls(benchmark=self.BENCHMARK_NAME)  # type: ignore[call-arg]
        pruned = pruner.prune(samples, prune_ratio, seed=seed)

        logger.info(
            "[%s] pruned %d → %d samples (strategy=%s, ratio=%.2f)",
            self.DATASET_ID, len(samples), len(pruned), strategy_name, prune_ratio,
        )
        return pruned

    # ------------------------------------------------------------------
    # Override in subclass
    # ------------------------------------------------------------------

    def _load_full_samples(self) -> list[dict[str, Any]]:
        raise NotImplementedError
