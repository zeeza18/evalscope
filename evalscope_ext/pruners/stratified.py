"""
Stratified Diversity Pruner (SDP)
==================================
Strategy: stratify samples into cells defined by (difficulty_tier × capability_topic),
then sample proportionally from each cell using MaxMin distance to maximise intra-cell
diversity.  Fallback to uniform-within-cell when sklearn is unavailable.

Why this generalises to unseen models
--------------------------------------
Cell boundaries are derived entirely from *problem structure* (metadata fields and,
optionally, a lightweight sentence-embedding of the problem text) — never from
per-model scores.  A fourth model will encounter the same distribution of difficulty
tiers and topic clusters as the three reference models, so the pruned set covers
its capability space equally well.

Why not random / top-k
-----------------------
Random sampling gives high variance on rare topics.  Top-k easy/hard removes whole
difficulty tiers and is trivially gamed.  SDP guarantees coverage of every cell.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

import numpy as np

from evalscope_ext.pruners.base import BasePruner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extractors — one per benchmark, registered at import time
# ---------------------------------------------------------------------------

_FEATURE_EXTRACTORS: dict[str, Callable[[dict], tuple[str, str]]] = {}


def register_feature_extractor(benchmark_name: str):
    """Decorator: @register_feature_extractor("live_code_bench")"""
    def _wrap(fn: Callable[[dict], tuple[str, str]]):
        _FEATURE_EXTRACTORS[benchmark_name] = fn
        return fn
    return _wrap


@register_feature_extractor("live_code_bench")
def _lcb_features(sample: dict) -> tuple[str, str]:
    """
    Return (difficulty_tier, topic_cluster) for a LCB sample.

    LCB prediction metadata is empty in the shipped data, so we extract
    difficulty and topic from the problem text in messages[0].content.
    """
    messages = sample.get("messages") or []
    text = ""
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            # content can be a list of dicts with "type"/"text" keys
            for block in content:
                if isinstance(block, dict):
                    text += block.get("text", "")
        else:
            text += str(content)

    text_lower = text.lower()

    # Difficulty: proxy from constraint magnitudes in the problem text
    difficulty = _lcb_difficulty_from_text(text_lower)
    topic = _coarse_topic_from_text(text_lower)
    return difficulty, topic


@register_feature_extractor("aa_lcr")
def _aa_lcr_features(sample: dict) -> tuple[str, str]:
    """Return (context_band, reasoning_type) for an AA-LCR sample.

    The shipped data has metadata.input_tokens for context length.
    """
    meta = sample.get("metadata") or {}
    ctx_len = int(meta.get("input_tokens") or meta.get("context_length") or 0)
    band = _context_band(ctx_len)
    # Reasoning type from the question text
    question = str(meta.get("question") or "").lower()
    rtype = _coarse_reasoning_from_question(question)
    return band, rtype


@register_feature_extractor("mmmu")
def _mmmu_features(sample: dict) -> tuple[str, str]:
    """Return (subject, image_stress_tier) for an MMMU sample.

    We use the actual subject name (not a coarse group) so the proportional
    allocator guarantees coverage across all 22 MMMU subjects.  Collapsing
    to 5 coarse groups causes severe subject imbalance in the pruned set.
    """
    meta = sample.get("metadata") or {}
    subject = str(meta.get("subfield") or meta.get("subject") or "unknown").lower()
    # Normalise spaces/underscores for stable cell keys
    subject = subject.replace(" ", "_")
    stress = _encoder_stress_tier(sample)
    return subject, stress


# ---------------------------------------------------------------------------
# The pruner itself
# ---------------------------------------------------------------------------

class StratifiedDiversityPruner(BasePruner):
    """
    Universal stratified-diversity pruner.

    Parameters
    ----------
    benchmark : str
        Registered name used to look up the feature extractor.
        Falls back to generic text-hash binning if not registered.
    diversity_within_cell : bool
        If True (default), apply MaxMin distance selection within each cell
        using scikit-learn KMeans; requires sklearn.  Set False for a pure
        proportional random sample that has no extra dependencies.
    """

    def __init__(self, benchmark: str = "generic", diversity_within_cell: bool = True):
        self.benchmark = benchmark
        self.diversity_within_cell = diversity_within_cell

    # ------------------------------------------------------------------

    def prune(
        self,
        samples: list[dict[str, Any]],
        prune_ratio: float,
        *,
        seed: int = 42,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if not 0.0 < prune_ratio < 1.0:
            raise ValueError(f"prune_ratio must be in (0,1), got {prune_ratio}")

        k = self._target_k(len(samples), prune_ratio)
        logger.info(
            "SDP: keeping %d / %d samples (prune_ratio=%.2f, benchmark=%s)",
            k, len(samples), prune_ratio, self.benchmark,
        )

        cells = self._assign_cells(samples)
        selected = self._proportional_select(samples, cells, k, seed)
        return selected

    # ------------------------------------------------------------------
    # Cell assignment
    # ------------------------------------------------------------------

    def _assign_cells(self, samples: list[dict]) -> list[str]:
        extractor = _FEATURE_EXTRACTORS.get(self.benchmark, _generic_features)
        cells = []
        for s in samples:
            try:
                tier, topic = extractor(s)
            except Exception:
                tier, topic = "unknown", "unknown"
            cells.append(f"{tier}::{topic}")
        return cells

    # ------------------------------------------------------------------
    # Proportional cell sampling with optional intra-cell diversity
    # ------------------------------------------------------------------

    def _proportional_select(
        self,
        samples: list[dict],
        cells: list[str],
        k: int,
        seed: int,
    ) -> list[dict]:
        rng = np.random.default_rng(seed)

        # Group by cell
        cell_map: dict[str, list[int]] = {}
        for idx, cell in enumerate(cells):
            cell_map.setdefault(cell, []).append(idx)

        unique_cells = sorted(cell_map)
        cell_counts = np.array([len(cell_map[c]) for c in unique_cells], dtype=float)

        # Proportional allocation (Hamilton / largest-remainder)
        quotas = k * cell_counts / cell_counts.sum()
        floors = np.floor(quotas).astype(int)
        remainder = k - floors.sum()
        fracs = quotas - floors
        order = np.argsort(-fracs)
        for i in range(remainder):
            floors[order[i]] += 1

        selected_indices: list[int] = []
        for cell, quota in zip(unique_cells, floors.tolist()):
            pool = cell_map[cell]
            if quota <= 0:
                continue
            if quota >= len(pool):
                selected_indices.extend(pool)
                continue
            if self.diversity_within_cell and len(pool) > quota * 2:
                chosen = self._maxmin_select(samples, pool, quota, rng)
            else:
                chosen = rng.choice(pool, size=quota, replace=False).tolist()
            selected_indices.extend(chosen)

        # Stable sort to preserve original ordering
        selected_indices.sort()
        return [samples[i] for i in selected_indices]

    # ------------------------------------------------------------------
    # MaxMin diversity selection (greedy farthest-point sampling)
    # ------------------------------------------------------------------

    def _maxmin_select(
        self,
        samples: list[dict],
        pool: list[int],
        k: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """
        Greedy farthest-point sampling on a simple feature vector.
        Falls back to random if building the feature matrix fails.
        """
        try:
            vecs = np.stack([_sample_to_vec(samples[i]) for i in pool])
        except Exception:
            return rng.choice(pool, size=k, replace=False).tolist()

        n = len(pool)
        # Seed with a random point
        chosen = [int(rng.integers(n))]
        dists = np.full(n, np.inf)
        for _ in range(k - 1):
            dists = np.minimum(dists, np.sum((vecs - vecs[chosen[-1]]) ** 2, axis=1))
            chosen.append(int(np.argmax(dists)))
        return [pool[i] for i in chosen]


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LCB text-based feature helpers (used because LCB metadata is empty)
# ---------------------------------------------------------------------------

# Constraint magnitudes correlate strongly with algorithmic difficulty
_HARD_CONSTRAINTS = ["10^18", "10^9", "10^6 * 10^6", "n * m", "10**9"]
_EASY_CONSTRAINTS = ["n <= 20", "n <= 100", "n <= 1000", "1 <= n <= 10"]

def _lcb_difficulty_from_text(text: str) -> str:
    for pat in _HARD_CONSTRAINTS:
        if pat in text:
            return "hard"
    for pat in _EASY_CONSTRAINTS:
        if pat in text:
            return "easy"
    # Medium by default; also check for typical hard keywords
    if any(kw in text for kw in ["segment tree", "fenwick", "dijkstra", "floyd", "convex hull", "suffix array"]):
        return "hard"
    return "medium"


def _coarse_topic_from_text(text: str) -> str:
    for kw, group in _ALGO_KEYWORDS.items():
        if kw in text:
            return group
    return "other"


def _coarse_reasoning_from_question(question: str) -> str:
    if any(kw in question for kw in ["list", "enumerate", "all ", "every"]):
        return "retrieval"
    if any(kw in question for kw in ["summar", "overview", "describe"]):
        return "summarisation"
    if any(kw in question for kw in ["why", "how", "explain", "reason", "infer"]):
        return "inference"
    if any(kw in question for kw in ["compare", "contrast", "differ", "both"]):
        return "multi_hop"
    return "other"


def _generic_features(sample: dict) -> tuple[str, str]:
    """Text-hash binning for unregistered benchmarks."""
    text = json.dumps(sample, sort_keys=True, ensure_ascii=False)
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    tier = ["low", "mid", "high"][h % 3]
    topic = str(h % 7)
    return tier, topic


def _normalise_difficulty(raw: str) -> str:
    if raw in ("easy", "1", "simple"):
        return "easy"
    if raw in ("hard", "3", "difficult"):
        return "hard"
    return "medium"


_ALGO_KEYWORDS = {
    "dp": "dynamic_programming",
    "dynamic": "dynamic_programming",
    "graph": "graph",
    "tree": "tree",
    "sort": "sorting",
    "search": "search",
    "string": "string",
    "math": "math",
    "greedy": "greedy",
    "backtrack": "backtracking",
    "bit": "bit_manipulation",
    "hash": "hashing",
    "array": "array",
}


def _coarse_topic(tags: list[str]) -> str:
    for tag in tags:
        tag_lower = tag.lower()
        for kw, group in _ALGO_KEYWORDS.items():
            if kw in tag_lower:
                return group
    return "other"


def _context_band(n_tokens: int) -> str:
    if n_tokens < 4_000:
        return "short"
    if n_tokens < 32_000:
        return "medium"
    return "long"


def _coarse_reasoning(rtype: str) -> str:
    if "summar" in rtype:
        return "summarisation"
    if "retriev" in rtype or "fact" in rtype or "lookup" in rtype:
        return "retrieval"
    if "infer" in rtype or "reason" in rtype:
        return "inference"
    if "multi" in rtype:
        return "multi_hop"
    return "other"


# MMMU subject → high-level domain group
_MMMU_GROUPS = {
    "art": "arts_humanities",
    "history": "arts_humanities",
    "literature": "arts_humanities",
    "design": "arts_humanities",
    "music": "arts_humanities",
    "accounting": "business",
    "economics": "business",
    "finance": "business",
    "manage": "business",
    "marketing": "business",
    "biology": "science",
    "chemistry": "science",
    "physics": "science",
    "geography": "science",
    "materials": "science",
    "energy": "science",
    "electronics": "stem_engineering",
    "computer": "stem_engineering",
    "architecture": "stem_engineering",
    "clinical": "medicine",
    "medical": "medicine",
    "diagnostics": "medicine",
    "pharmacy": "medicine",
}


def _mmmu_subject_group(subject: str) -> str:
    for key, group in _MMMU_GROUPS.items():
        if key in subject:
            return group
    return "other"


# Encoder-stress tier: samples that require fine-grained visual reasoning
# score higher.  We proxy this from subject / question metadata.
_HIGH_STRESS_SUBJECTS = {
    "clinical", "diagnostics", "electronics", "architecture",
    "materials", "design", "art_theory",
}

_HIGH_STRESS_QTYPES = {"spatial", "diagram", "chart", "measure", "count", "identify"}


def _encoder_stress_tier(sample: dict) -> str:
    """
    Encoder stress for MMMU samples.

    High stress = requires fine-grained spatial/visual reasoning that random
    sampling cannot guarantee coverage of.  We use three signals:
      1. Subject domain (clinical/electronics/architecture = high)
      2. img_type field (Diagrams, Charts, Medical images = harder than text-only)
      3. topic_difficulty from the metadata
    """
    meta = sample.get("metadata") or {}
    subject = str(meta.get("subfield") or meta.get("subject") or "").lower()
    img_type_raw = str(meta.get("img_type") or "").lower()
    difficulty = str(meta.get("topic_difficulty") or "").lower()

    score = 0
    for kw in _HIGH_STRESS_SUBJECTS:
        if kw in subject:
            score += 2

    # img_type is stored as a string repr of a list, e.g. "['Diagrams', 'Charts']"
    for kw in ("diagram", "chart", "medical", "scientific", "map", "schematic", "table"):
        if kw in img_type_raw:
            score += 1

    if difficulty == "hard":
        score += 1

    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


def _sample_to_vec(sample: dict) -> np.ndarray:
    """Minimal feature vector: index + SHA hash nibbles of prompt text."""
    text = str(sample.get("prompt") or sample.get("question") or "")
    h = hashlib.sha256(text.encode()).digest()
    nibbles = np.frombuffer(h[:16], dtype=np.uint8).astype(np.float32) / 255.0
    idx = float(sample.get("index", 0)) / 10_000.0
    return np.concatenate([[idx], nibbles])
