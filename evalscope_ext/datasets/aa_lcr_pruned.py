"""
evalscope adapter: aa_lcr_pruned
=================================
Same pattern as live_code_bench_pruned — stratified-diversity pruning
applied to the AA-LCR long-context reasoning benchmark.

Usage:
    evalscope eval \\
        --model <model> \\
        --datasets aa_lcr_pruned \\
        --dataset-args '{"pruning_strategy": "stratified_diversity", "prune_ratio": 0.5}' \\
        --output ./results_pruned/
"""
from __future__ import annotations

from typing import Any

from evalscope_ext.datasets._base import PrunedDatasetAdapter


class AaLcrPruned(PrunedDatasetAdapter):
    """Pruned AA-LCR — long-context reasoning benchmark."""

    BENCHMARK_NAME = "aa_lcr"
    DATASET_ID = "aa_lcr_pruned"
    SCORE_FIELD = "acc"

    def _load_full_samples(self) -> list[dict[str, Any]]:
        from evalscope.benchmarks.aa_lcr import AaLcr  # type: ignore
        base = AaLcr(self.model_args)
        return list(base.load_dataset())


try:
    from evalscope.benchmarks import register_benchmark  # type: ignore
    register_benchmark("aa_lcr_pruned", AaLcrPruned)
except Exception:
    pass
