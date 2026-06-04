"""
compare_runs — compare a full eval run against its pruned counterpart.

Usage:
    python -m evalscope_ext.tools.compare_runs \\
        --full  ./results_full/ \\
        --pruned ./results_pruned/

Output: a Markdown-formatted table + Pearson r between pruned and full
scores, plus a go/no-go signal per model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_results(directory: Path) -> dict[str, dict[str, float]]:
    """
    Return {model_name: {benchmark: score}} by scanning evalscope output dirs.

    evalscope writes results/<model>/<benchmark>/result.json (or similar).
    We accept multiple layouts and fall back to scanning for any *.json.
    """
    results: dict[str, dict[str, float]] = {}
    for model_dir in sorted(directory.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        results[model] = {}
        for bench_dir in sorted(model_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            bench = bench_dir.name
            score = _extract_score(bench_dir)
            if score is not None:
                results[model][bench] = score
    return results


def _extract_score(bench_dir: Path) -> float | None:
    """Find the aggregate score in any *.json inside bench_dir."""
    candidates = sorted(bench_dir.glob("*.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # evalscope usually writes {"score": x} or {"accuracy": x}
            for key in ("score", "accuracy", "pass", "acc", "pass@1"):
                if key in data:
                    return float(data[key])
            # Sometimes nested under "results"
            if "results" in data and isinstance(data["results"], dict):
                for v in data["results"].values():
                    if isinstance(v, (int, float)):
                        return float(v)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return num / (sx * sy)


def _go_nogo(full_score: float, threshold: float = 0.6) -> str:
    return "GO  ✓" if full_score >= threshold else "NO-GO ✗"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _render_report(
    full: dict[str, dict[str, float]],
    pruned: dict[str, dict[str, float]],
    go_threshold: float,
) -> str:
    lines: list[str] = []
    lines.append("# Pruned vs Full Benchmark Comparison\n")

    all_models = sorted(set(full) | set(pruned))
    all_benches = sorted(
        {b for scores in full.values() for b in scores}
        | {b for scores in pruned.values() for b in scores}
    )

    # Per-model, per-benchmark table
    header = "| Model | Benchmark | Full | Pruned | Δ | Go/No-Go |"
    sep = "|---|---|---:|---:|---:|---|"
    lines.append(header)
    lines.append(sep)

    paired_full: list[float] = []
    paired_pruned: list[float] = []

    for model in all_models:
        for bench in all_benches:
            fs = full.get(model, {}).get(bench)
            ps = pruned.get(model, {}).get(bench)
            if fs is None and ps is None:
                continue
            fs_str = f"{fs:.3f}" if fs is not None else "—"
            ps_str = f"{ps:.3f}" if ps is not None else "—"
            delta = f"{ps - fs:+.3f}" if (fs is not None and ps is not None) else "—"
            signal = _go_nogo(fs, go_threshold) if fs is not None else "—"
            lines.append(f"| {model} | {bench} | {fs_str} | {ps_str} | {delta} | {signal} |")
            if fs is not None and ps is not None:
                paired_full.append(fs)
                paired_pruned.append(ps)

    lines.append("")

    # Correlation summary
    r = _pearson(paired_full, paired_pruned)
    lines.append(f"**Pearson r (pruned ↔ full):** {r:.4f}  _(n={len(paired_full)} pairs)_\n")
    if not math.isnan(r):
        if r >= 0.95:
            lines.append("> Pruned scores are a reliable proxy for full-run scores.")
        elif r >= 0.80:
            lines.append("> Pruned scores are a reasonable proxy; verify borderline models.")
        else:
            lines.append("> Low correlation — consider a higher prune_ratio or a different strategy.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare full vs pruned evalscope runs."
    )
    parser.add_argument("--full", required=True, type=Path, help="Full-run results directory")
    parser.add_argument("--pruned", required=True, type=Path, help="Pruned-run results directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="Go/no-go score threshold (default 0.6)")
    parser.add_argument("--out", type=Path, default=None, help="Write report to this file (default: stdout)")
    args = parser.parse_args(argv)

    full = _load_results(args.full)
    pruned = _load_results(args.pruned)

    if not full and not pruned:
        print(
            "WARNING: no result JSON files found under either directory.\n"
            "If running against the shipped JSONL data, use run_analysis.py instead.",
            file=sys.stderr,
        )

    report = _render_report(full, pruned, args.threshold)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
