"""
Stratified Diversity Pruner (SDP)
==================================
Partitions samples into cells defined by two structural axes
(e.g. difficulty × topic for LCB, context-band × reasoning-type for AA-LCR),
allocates a proportional quota to each cell via Hamilton's method, then
selects within each cell using greedy MaxMin farthest-point sampling to
maximise intra-cell diversity.

Why this generalises to an unseen model
----------------------------------------
Cell boundaries are derived from *problem structure* — metadata fields,
constraint patterns, keyword presence — never from per-model scores.
A fourth model encounters the same difficulty-and-topic distribution as the
three reference models, so the pruned set covers its capability space equally.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

import numpy as np

from evalscope.pruners.base import BasePruner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-extractor registry  (benchmark_name -> fn(sample) -> (str, str))
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, Callable[[Any], tuple[str, str]]] = {}


def register_extractor(benchmark: str):
    """Decorator: ``@register_extractor('live_code_bench')``"""
    def _wrap(fn: Callable[[Any], tuple[str, str]]):
        _EXTRACTORS[benchmark] = fn
        return fn
    return _wrap


# ---------------------------------------------------------------------------
# Feature extractors — one per benchmark
# ---------------------------------------------------------------------------

@register_extractor('live_code_bench')
def _lcb_features(sample: Any) -> tuple[str, str]:
    """(difficulty_tier, topic_cluster) for a LiveCodeBench Sample.

    LCB metadata from evalscope only carries ``contest_date``; we use it
    as a temporal difficulty proxy (older = easier, newer = harder) and
    extract algorithmic topic from the prompt text.
    """
    meta = getattr(sample, 'metadata', {}) or {}
    date_str = str(meta.get('contest_date', '') or '')
    difficulty = _date_to_difficulty(date_str)

    # Input is a list of ChatMessage objects; pull text from content
    text = _extract_text(sample)
    topic = _topic_from_text(text.lower())
    return difficulty, topic


@register_extractor('aa_lcr')
def _aa_lcr_features(sample: Any) -> tuple[str, str]:
    """(context_band, reasoning_type) for an AA-LCR Sample."""
    meta = getattr(sample, 'metadata', {}) or {}
    tokens = int(meta.get('input_tokens', 0) or 0)
    band = _context_band(tokens)
    question = str(meta.get('question', '') or '').lower()
    rtype = _reasoning_from_question(question)
    return band, rtype


@register_extractor('mmmu')
def _mmmu_features(sample: Any) -> tuple[str, str]:
    """(subject, encoder_stress_tier) for an MMMU Sample.

    We keep subject granular (22 unique values) so the proportional
    allocator guarantees per-subject coverage.  Collapsing to 5 coarse
    domain groups causes severe subject imbalance.
    """
    meta = getattr(sample, 'metadata', {}) or {}
    subfield = str(meta.get('subfield', '') or '').lower().replace(' ', '_')
    subject = subfield if subfield else 'unknown'
    stress = _encoder_stress(meta)
    return subject, stress


# ---------------------------------------------------------------------------
# The pruner
# ---------------------------------------------------------------------------

class StratifiedDiversityPruner(BasePruner):
    """Stratified-diversity pruner — works on any evalscope ``Sample`` list.

    Parameters
    ----------
    benchmark : str
        Key into the feature-extractor registry.  Use ``'generic'`` for
        benchmarks without a registered extractor.
    diversity_within_cell : bool
        If ``True`` (default), apply MaxMin farthest-point sampling inside
        each cell.  Set ``False`` for pure proportional random sampling
        (no extra dependencies, slightly worse coverage).
    """

    def __init__(self, benchmark: str = 'generic', diversity_within_cell: bool = True):
        self.benchmark = benchmark
        self.diversity_within_cell = diversity_within_cell

    def prune(self, samples: list[Any], prune_ratio: float, *, seed: int = 42) -> list[Any]:
        if not 0.0 < prune_ratio < 1.0:
            raise ValueError(f'prune_ratio must be in (0, 1), got {prune_ratio}')

        k = self.target_k(len(samples), prune_ratio)
        logger.info(
            'SDP[%s]: %d → %d samples (prune_ratio=%.2f)',
            self.benchmark, len(samples), k, prune_ratio,
        )

        cells = self._assign_cells(samples)
        return self._proportional_select(samples, cells, k, seed)

    # ------------------------------------------------------------------

    def _assign_cells(self, samples: list[Any]) -> list[str]:
        extractor = _EXTRACTORS.get(self.benchmark, _generic_features)
        out = []
        for s in samples:
            try:
                a, b = extractor(s)
            except Exception:
                a, b = 'unknown', 'unknown'
            out.append(f'{a}::{b}')
        return out

    def _proportional_select(
        self, samples: list[Any], cells: list[str], k: int, seed: int
    ) -> list[Any]:
        rng = np.random.default_rng(seed)

        cell_map: dict[str, list[int]] = {}
        for i, c in enumerate(cells):
            cell_map.setdefault(c, []).append(i)

        unique = sorted(cell_map)
        counts = np.array([len(cell_map[c]) for c in unique], dtype=float)
        quotas = k * counts / counts.sum()
        floors = np.floor(quotas).astype(int)
        remainder = k - floors.sum()
        for i in np.argsort(-(quotas - floors))[:remainder]:
            floors[i] += 1

        chosen: list[int] = []
        for cell, quota in zip(unique, floors.tolist()):
            pool = cell_map[cell]
            if quota <= 0:
                continue
            if quota >= len(pool):
                chosen.extend(pool)
                continue
            if self.diversity_within_cell and len(pool) > quota * 2:
                chosen.extend(self._maxmin(samples, pool, quota, rng))
            else:
                chosen.extend(rng.choice(pool, size=quota, replace=False).tolist())

        chosen.sort()
        return [samples[i] for i in chosen]

    def _maxmin(
        self, samples: list[Any], pool: list[int], k: int, rng: np.random.Generator
    ) -> list[int]:
        """Greedy farthest-point sampling on a lightweight text-hash feature vector."""
        try:
            vecs = np.stack([_to_vec(samples[i]) for i in pool])
        except Exception:
            return rng.choice(pool, size=k, replace=False).tolist()

        n = len(pool)
        chosen = [int(rng.integers(n))]
        dists = np.full(n, np.inf)
        for _ in range(k - 1):
            dists = np.minimum(dists, np.sum((vecs - vecs[chosen[-1]]) ** 2, axis=1))
            chosen.append(int(np.argmax(dists)))
        return [pool[i] for i in chosen]


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _extract_text(sample: Any) -> str:
    """Pull plain text out of a Sample.input (str or list of ChatMessages)."""
    inp = getattr(sample, 'input', '') or ''
    if isinstance(inp, str):
        return inp
    text = []
    for msg in inp:
        content = getattr(msg, 'content', '') or ''
        if isinstance(content, str):
            text.append(content)
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, 'text'):
                    text.append(block.text or '')
                elif isinstance(block, dict):
                    text.append(block.get('text', ''))
    return ' '.join(text)


def _date_to_difficulty(date_str: str) -> str:
    """Map contest_date to a difficulty proxy.

    LCB problems are ordered chronologically; problems from earlier
    contest periods (2022) tend to appear in training data and are
    easier for current models; 2024+ problems are harder.
    """
    if not date_str:
        return 'medium'
    year = date_str[:4]
    if year <= '2022':
        return 'easy'
    if year >= '2024':
        return 'hard'
    return 'medium'


_ALGO_KEYWORDS: dict[str, str] = {
    'dynamic programming': 'dp',
    ' dp ': 'dp',
    'graph': 'graph',
    'tree': 'tree',
    'sort': 'sorting',
    'binary search': 'search',
    'greedy': 'greedy',
    'string': 'string',
    'math': 'math',
    'backtrack': 'backtracking',
    'bit manipulation': 'bit',
    'hash': 'hashing',
    'array': 'array',
    'stack': 'stack',
    'queue': 'queue',
    'heap': 'heap',
    'trie': 'trie',
    'segment tree': 'segment_tree',
}


def _topic_from_text(text: str) -> str:
    for kw, group in _ALGO_KEYWORDS.items():
        if kw in text:
            return group
    return 'other'


def _context_band(tokens: int) -> str:
    if tokens < 8_000:
        return 'short'
    if tokens < 64_000:
        return 'medium'
    return 'long'


def _reasoning_from_question(q: str) -> str:
    if any(kw in q for kw in ('list', 'enumerate', 'all ', 'every', 'which')):
        return 'retrieval'
    if any(kw in q for kw in ('summar', 'overview', 'describe')):
        return 'summarisation'
    if any(kw in q for kw in ('compare', 'contrast', 'differ', 'both')):
        return 'multi_hop'
    if any(kw in q for kw in ('why', 'how', 'explain', 'reason', 'infer')):
        return 'inference'
    return 'other'


_HIGH_STRESS_SUBJECTS = {
    'electronics', 'architecture_and_engineering', 'clinical_medicine',
    'diagnostics_and_laboratory_medicine', 'materials', 'energy_and_power',
    'basic_medical_science',
}

_HIGH_STRESS_IMG_TYPES = {'diagram', 'scientific', 'medical', 'chart', 'schematic', 'map'}


def _encoder_stress(meta: dict) -> str:
    """Three-tier encoder-stress score for MMMU samples.

    High = questions where visual information is load-bearing and the
    image encodes spatial / fine-grained structure (diagrams, medical
    images, engineering schematics).
    """
    subfield = str(meta.get('subfield', '') or '').lower().replace(' ', '_')
    img_type = str(meta.get('img_type', '') or '').lower()
    difficulty = str(meta.get('topic_difficulty', '') or '').lower()

    score = 0
    if subfield in _HIGH_STRESS_SUBJECTS:
        score += 2
    for kw in _HIGH_STRESS_IMG_TYPES:
        if kw in img_type:
            score += 1
    if difficulty == 'hard':
        score += 1

    if score >= 3:
        return 'high'
    if score >= 1:
        return 'medium'
    return 'low'


def _generic_features(sample: Any) -> tuple[str, str]:
    """Hash-based binning for benchmarks without a registered extractor."""
    text = _extract_text(sample)
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return str(h % 3), str(h % 7)


def _to_vec(sample: Any) -> np.ndarray:
    """Lightweight feature vector for MaxMin distance computation."""
    text = _extract_text(sample)
    h = hashlib.sha256(text.encode()).digest()
    nibbles = np.frombuffer(h[:16], dtype=np.uint8).astype(np.float32) / 255.0
    sample_id = float(getattr(sample, 'id', None) or 0) / 10_000.0
    return np.concatenate([[sample_id], nibbles])
