# flake8: noqa: E501
"""
mmmu_pruned — stratified-diversity pruned MMMU adapter.

Stratifies on (subject × encoder_stress_tier), ensuring every MMMU subject
retains ~equal representation while over-weighting questions that stress the
image encoder specifically (diagrams, medical images, engineering schematics).

CLI usage::

    evalscope eval \\
        --model <model> \\
        --datasets mmmu_pruned \\
        --dataset-args '{"prune_ratio": 0.5}' \\
        --output ./results_pruned/
"""
from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MmmuAdapter
from evalscope.constants import Tags
from evalscope.pruners import StratifiedDiversityPruner
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (Pruned)',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=(
            'Stratified-diversity pruned MMMU.  '
            'Stratifies on subject × encoder-stress tier (derived from img_type '
            'and topic_difficulty metadata), ensuring per-subject coverage and '
            'over-weighting visually demanding questions. '
            'Configure via ``--dataset-args \'{"prune_ratio": 0.5}\'``.'
        ),
        dataset_id='MMMU/MMMU',
        metric_list=['acc'],
        eval_split='validation',
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
class MmmuPrunedAdapter(MmmuAdapter):
    """MMMU with stratified-diversity pruning applied before evaluation.

    Pruning is applied per-subset (each MMMU subject is a subset), then
    the results are re-assembled into a pruned DatasetDict.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._prune_ratio: float = float(self.extra_params.get('prune_ratio', 0.5))
        self._prune_seed: int = int(self.extra_params.get('prune_seed', 42))

    def load_dataset(self) -> DatasetDict:
        dataset_dict = super().load_dataset()
        pruner = StratifiedDiversityPruner(benchmark='mmmu')
        total_before, total_after = 0, 0
        for subset_name in list(dataset_dict.keys()):
            original = list(dataset_dict[subset_name])
            kept = pruner.prune(original, self._prune_ratio, seed=self._prune_seed)
            total_before += len(original)
            total_after += len(kept)
            dataset_dict[subset_name] = MemoryDataset(
                samples=kept,
                name=f'{subset_name}_pruned',
                location=dataset_dict[subset_name].location,
            )
        logger.info(
            'mmmu_pruned: %d → %d samples across %d subjects',
            total_before, total_after, len(dataset_dict),
        )
        return dataset_dict
