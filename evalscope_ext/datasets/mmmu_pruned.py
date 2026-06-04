"""
evalscope adapter: mmmu_pruned
================================
Stratified-diversity pruner for MMMU, weighted toward subjects that
stress image encoders specifically (see pruners/stratified.py for
the encoder-stress tier logic).

Usage:
    evalscope eval \\
        --model <model> \\
        --datasets mmmu_pruned \\
        --dataset-args '{"pruning_strategy": "stratified_diversity", "prune_ratio": 0.5}' \\
        --output ./results_pruned/
"""
from __future__ import annotations

from typing import Any

from evalscope_ext.datasets._base import PrunedDatasetAdapter


class MmmuPruned(PrunedDatasetAdapter):
    """Pruned MMMU — multimodal image-encoder probe."""

    BENCHMARK_NAME = "mmmu"
    DATASET_ID = "mmmu_pruned"
    SCORE_FIELD = "acc"

    def _load_full_samples(self) -> list[dict[str, Any]]:
        from evalscope.benchmarks.mmmu import MMMU  # type: ignore
        base = MMMU(self.model_args)
        return list(base.load_dataset())


try:
    from evalscope.benchmarks import register_benchmark  # type: ignore
    register_benchmark("mmmu_pruned", MmmuPruned)
except Exception:
    pass
