# flake8: noqa: E501
"""
aa_lcr_pruned — stratified-diversity pruned AA-LCR adapter.

Stratifies on (context_length_band × reasoning_type) using the
``input_tokens`` field in sample metadata.

CLI usage::

    evalscope eval \\
        --model <model> \\
        --datasets aa_lcr_pruned \\
        --dataset-args '{"prune_ratio": 0.5}' \\
        --output ./results_pruned/
"""
from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AaLcrAdapter
from evalscope.constants import Tags
from evalscope.pruners import StratifiedDiversityPruner
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description=(
            'Stratified-diversity pruned AA-LCR.  '
            'Stratifies on context-length band × reasoning type using the '
            '``input_tokens`` field. '
            'Configure via ``--dataset-args \'{"prune_ratio": 0.5}\'``.'
        ),
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        eval_split='test',
        extra_params={
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to remove (0 < prune_ratio < 1).',
                'value': 0.5,
            },
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy name.',
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
class AaLcrPrunedAdapter(AaLcrAdapter):
    """AA-LCR with stratified-diversity pruning applied before evaluation."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._prune_ratio: float = float(self.extra_params.get('prune_ratio', 0.5))
        self._prune_seed: int = int(self.extra_params.get('prune_seed', 42))

    def load_dataset(self) -> DatasetDict:
        dataset_dict = super().load_dataset()
        pruner = StratifiedDiversityPruner(benchmark='aa_lcr')
        for subset_name in list(dataset_dict.keys()):
            original = list(dataset_dict[subset_name])
            kept = pruner.prune(original, self._prune_ratio, seed=self._prune_seed)
            logger.info(
                'aa_lcr_pruned[%s]: %d → %d samples',
                subset_name, len(original), len(kept),
            )
            dataset_dict[subset_name] = MemoryDataset(
                samples=kept,
                name=f'{subset_name}_pruned',
                location=dataset_dict[subset_name].location,
            )
        return dataset_dict
