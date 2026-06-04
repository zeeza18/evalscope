# flake8: noqa: E501
"""
live_code_bench_pruned — stratified-diversity pruned LiveCodeBench adapter.

Registers the ``live_code_bench_pruned`` benchmark name in evalscope.
All eval logic (sandboxed execution, pass@k scoring) is inherited from
``LiveCodeBenchAdapter``; this subclass only prunes the dataset before
evaluation begins.

CLI usage::

    evalscope eval \\
        --model <model> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"prune_ratio": 0.5, "pruning_strategy": "stratified_diversity"}' \\
        --output ./results_pruned/
"""
from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope.pruners import StratifiedDiversityPruner
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description=(
            'Stratified-diversity pruned LiveCodeBench v5.  '
            'Preserves the full difficulty × topic distribution at a fraction of the cost. '
            'Configure via ``--dataset-args \'{"prune_ratio": 0.5}\'``.'
        ),
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        extra_params={
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to remove (0 < prune_ratio < 1).',
                'value': 0.5,
            },
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy name (currently only "stratified_diversity").',
                'value': 'stratified_diversity',
            },
            'prune_seed': {
                'type': 'int',
                'description': 'RNG seed for reproducibility.',
                'value': 42,
            },
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """LiveCodeBench with stratified-diversity pruning applied before evaluation."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._prune_ratio: float = float(self.extra_params.get('prune_ratio', 0.5))
        self._prune_seed: int = int(self.extra_params.get('prune_seed', 42))

    def load_dataset(self) -> DatasetDict:
        dataset_dict = super().load_dataset()
        pruner = StratifiedDiversityPruner(benchmark='live_code_bench')
        for subset_name in list(dataset_dict.keys()):
            original = list(dataset_dict[subset_name])
            kept = pruner.prune(original, self._prune_ratio, seed=self._prune_seed)
            logger.info(
                'live_code_bench_pruned[%s]: %d → %d samples',
                subset_name, len(original), len(kept),
            )
            dataset_dict[subset_name] = MemoryDataset(
                samples=kept,
                name=f'{subset_name}_pruned',
                location=dataset_dict[subset_name].location,
            )
        return dataset_dict
