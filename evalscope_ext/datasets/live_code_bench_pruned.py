"""
evalscope adapter: live_code_bench_pruned
=========================================
Drop-in replacement for the built-in LiveCodeBench adapter that applies
StratifiedDiversityPruner before returning samples.

Usage in evalscope CLI:
    evalscope eval \\
        --model <model> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"pruning_strategy": "stratified_diversity", "prune_ratio": 0.5}' \\
        --output ./results_pruned/
"""
from __future__ import annotations

from typing import Any

from evalscope_ext.datasets._base import PrunedDatasetAdapter


class LiveCodeBenchPruned(PrunedDatasetAdapter):
    """Pruned LiveCodeBench v5 — coding capability benchmark."""

    BENCHMARK_NAME = "live_code_bench"
    DATASET_ID = "live_code_bench_pruned"
    SCORE_FIELD = "pass"

    def _load_full_samples(self) -> list[dict[str, Any]]:
        # Delegate to the parent evalscope LiveCodeBench loader
        from evalscope.benchmarks.live_code_bench import LiveCodeBench  # type: ignore
        base = LiveCodeBench(self.model_args)
        return list(base.load_dataset())


# Self-register with evalscope's benchmark registry
try:
    from evalscope.benchmarks import register_benchmark  # type: ignore
    register_benchmark("live_code_bench_pruned", LiveCodeBenchPruned)
except Exception:
    # evalscope not installed yet — fine during development
    pass
