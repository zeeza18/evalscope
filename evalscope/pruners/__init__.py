"""
evalscope.pruners — benchmark sample pruning strategies.

Usage from an adapter::

    from evalscope.pruners import StratifiedDiversityPruner

    pruner = StratifiedDiversityPruner(benchmark='live_code_bench')
    kept: list[Sample] = pruner.prune(list(dataset), prune_ratio=0.5)
"""
from evalscope.pruners.base import BasePruner
from evalscope.pruners.stratified import StratifiedDiversityPruner

__all__ = ['BasePruner', 'StratifiedDiversityPruner']
