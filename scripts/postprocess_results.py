#!/usr/bin/env python3
"""Filter result JSONLs, validate coverage, and emit metrics summaries."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import BiGGenDatasetLoader, DatasetBundle, create_dataset_loader
from metrics import analyze_pointwise_results

DATASET_ROOTS: Mapping[str, Path] = {
    "biggen_bench": PROJECT_ROOT / "prometheus-eval" / "BiGGen-Bench",
    "llm_grader": PROJECT_ROOT / "llm_grader" / "dataset_os",
    "flask": PROJECT_ROOT / "FLASK",
    "mt_bench": PROJECT_ROOT / "prometheus-eval" / "MT-Bench",
    "chatbot_arena": PROJECT_ROOT / "prometheus-eval" / "ChatbotArena",
}

DEFAULT_GLOB = "results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-*.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "biggen_bench" / "filtered_blacklist"


@dataclass
class FilterStats:
    kept: int = 0
    skipped_tasks: int = 0
    skipped_range: int = 0

    def __iadd__(self, other: "FilterStats") -> "FilterStats":
        self.kept += other.kept
        self.skipped_tasks += other.skipped_tasks
        self.skipped_range += other.skipped_range
        return self


def _iter_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        pat_path = Path(pattern)
        if pat_path.is_absolute():
            matches = sorted(Path(p) for p in glob.glob(pattern))
        else:
            matches = sorted(PROJECT_ROOT.glob(pattern))
        paths.extend(matches)
    return paths


def _within(value: object, minimum: float | None, maximum: float | None) -> bool:
    if not isinstance(value, (int, float)):
        return True
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def _scores_within_bounds(record: dict, minimum: float | None, maximum: float | None) -> bool:
    if minimum is None and maximum is None:
        return True

    if not _within((record.get("result") or {}).get("score"), minimum, maximum):
        return False

    metadata = record.get("metadata") or {}
    perturbations = metadata.get("perturbations") or []
    if isinstance(perturbations, list):
        for entry in perturbations:
            if not isinstance(entry, dict):
                continue
            if not _within(entry.get("score"), minimum, maximum):
                return False
    perturb_scores = metadata.get("perturbation_scores") or {}
    if isinstance(perturb_scores, dict):
        for val in perturb_scores.values():
            if not _within(val, minimum, maximum):
                return False
    return True


def filter_file(
    src: Path,
    dst: Path,
    blacklist: set[str],
    *,
    min_score: float | None,
    max_score: float | None,
) -> FilterStats:
    stats = FilterStats()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            task = (record.get("metadata") or {}).get("task")
            if blacklist and task in blacklist:
                stats.skipped_tasks += 1
                continue
            if not _scores_within_bounds(record, min_score, max_score):
                stats.skipped_range += 1
                continue
            fout.write(line)
            stats.kept += 1
    return stats


def load_bundle(dataset: str, dataset_root: Path | None) -> DatasetBundle:
    if dataset_root is None:
        try:
            dataset_root = DATASET_ROOTS[dataset]
        except KeyError as exc:
            raise ValueError(
                f"No dataset root configured for '{dataset}'. Provide --dataset-root."
            ) from exc
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    loader = create_dataset_loader(dataset, dataset_root)
    return loader.load()


def validate_results(
    dataset: str,
    results_path: Path,
    *,
    min_ratio: float,
    min_count: int,
    dataset_root: Path | None,
) -> None:
    bundle = load_bundle(dataset, dataset_root)
    expected = len(bundle.entries)
    seen: set[str] = set()
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id") or record.get("data", {}).get("id")
            if record_id is not None:
                seen.add(str(record_id))
    actual = len(seen)
    ratio = actual / expected if expected else 0.0
    print(
        f"[validate] {results_path.name}: expected={expected} actual={actual} ratio={ratio:.4f} "
        f"(min_ratio={min_ratio})"
    )
    if actual < min_count:
        raise SystemExit(
            f"Validation failed for {results_path}: only {actual} results (< min_count {min_count})."
        )
    if expected and ratio < min_ratio:
        raise SystemExit(
            f"Validation failed for {results_path}: ratio {ratio:.4f} < threshold {min_ratio:.4f}."
        )


def summarize_metrics(results_path: Path) -> None:
    sections = analyze_pointwise_results(results_path)
    summary = sections.get("summary", {})
    distances = sections.get("distance_from_original", {})
    print(
        f"[metrics] {results_path.name}: examples={summary.get('examples_in_file')} "
        f"score_min={summary.get('score_min')} score_max={summary.get('score_max')} "
        f"MAD_O={distances.get('mean_MAD_O')} RMSD_O={distances.get('mean_RMSD_O')}"
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[DEFAULT_GLOB],
        help="Glob(s) for JSONL result files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for filtered JSONLs (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        default="biggen_bench",
        help="Dataset key for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override dataset root directory.",
    )
    parser.add_argument(
        "--blacklist-file",
        type=Path,
        help="Optional file listing task IDs to skip (one per line).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum acceptable numeric score (default: %(default)s)",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=5.0,
        help="Maximum acceptable numeric score (default: %(default)s)",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.95,
        help="Minimum coverage ratio for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum absolute count for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip metrics computation.",
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Skip validation step.",
    )
    return parser.parse_args(argv)


def load_blacklist(args: argparse.Namespace) -> set[str]:
    if args.blacklist_file:
        return {
            line.strip()
            for line in args.blacklist_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if args.dataset == "biggen_bench":
        return set(BiGGenDatasetLoader.TASK_BLACKLIST.keys())
    return set()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dataset_root = args.dataset_root.resolve() if args.dataset_root else None
    inputs = _iter_inputs(args.inputs)
    if not inputs:
        print("No input files matched the provided glob(s).", file=sys.stderr)
        return 1

    blacklist = load_blacklist(args)
    aggregate = FilterStats()

    for src in inputs:
        if not src.exists():
            print(f"[warn] skipping missing file {src}")
            continue
        dst = args.output_dir / src.name
        stats = filter_file(
            src,
            dst,
            blacklist,
            min_score=args.min_score,
            max_score=args.max_score,
        )
        aggregate += stats
        try:
            rel_dst = dst.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_dst = dst
        print(
            f"[filter] {src.name} -> {rel_dst} "
            f"(kept={stats.kept} skipped_tasks={stats.skipped_tasks} "
            f"skipped_range={stats.skipped_range})"
        )
        if not args.skip_validator:
            validate_results(
                args.dataset,
                dst,
                min_ratio=args.min_ratio,
                min_count=args.min_count,
                dataset_root=dataset_root,
            )
        if not args.skip_metrics:
            summarize_metrics(dst)

    print(
        f"[done] total kept={aggregate.kept} skipped_tasks={aggregate.skipped_tasks} "
        f"skipped_range={aggregate.skipped_range}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
