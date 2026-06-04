"""
Register all pruned dataset adapters with evalscope's benchmark registry.

Called once at package import time.
"""
from __future__ import annotations


def register_all() -> None:
    """Import each adapter module; side-effect registers with evalscope."""
    from evalscope_ext.datasets import live_code_bench_pruned  # noqa: F401
    from evalscope_ext.datasets import aa_lcr_pruned            # noqa: F401
    from evalscope_ext.datasets import mmmu_pruned              # noqa: F401
