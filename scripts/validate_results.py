#!/usr/bin/env python3
"""Validate that result files contain scores for the expected number of items."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import DatasetBundle, DatasetLoader, create_dataset_loader  # noqa: E402

DATASET_ROOTS: Dict[str, Path] = {
    "llm_grader": PROJECT_ROOT / "llm_grader" / "dataset_os",
    "biggen_bench": PROJECT_ROOT / "prometheus-eval" / "BiGGen-Bench",
    "flask": PROJECT_ROOT / "FLASK",
}

PLACEHOLDER_RESPONSES = {"hello world"}


def load_dataset(dataset: str, max_examples: int | None = None) -> DatasetBundle:
    try:
        root = DATASET_ROOTS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Known datasets: {', '.join(sorted(DATASET_ROOTS))}"
        ) from exc
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    loader: DatasetLoader = create_dataset_loader(dataset, root, max_examples=max_examples)
    return loader.load()


def _normalize_candidate(text: str) -> str:
    normalized = text.strip().lower()
    # Drop common trailing punctuation to make placeholder matching resilient.
    normalized = normalized.rstrip('!.?')
    return normalized


def _is_placeholder_response(record: Mapping[str, object]) -> bool:
    """Return True if the record contains a known dummy response."""

    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        student_answer = metadata.get("student_answer")
        if isinstance(student_answer, str):
            if _normalize_candidate(student_answer) in PLACEHOLDER_RESPONSES:
                return True

    data = record.get("data")
    if isinstance(data, Mapping):
        candidate = data.get("output")
        if isinstance(candidate, str):
            if _normalize_candidate(candidate) in PLACEHOLDER_RESPONSES:
                return True

    return False


def count_result_rows(results_path: Path) -> tuple[int, int]:
    if not results_path.exists():
        raise FileNotFoundError(f"Result file not found: {results_path}")
    seen_ids = set()
    filtered = 0
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id")
            if record_id is None:
                continue
            if _is_placeholder_response(record):
                filtered += 1
                continue
            seen_ids.add(record_id)
    return len(seen_ids), filtered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate that a result JSONL file covers most of the dataset."
    )
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g., llm_grader).")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to the aggregated result JSONL file (from results/…).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Override the expected dataset size (useful for smoke tests).",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.95,
        help="Minimum acceptable ratio of results to dataset size (default: 0.95).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum number of results required regardless of ratio.",
    )
    args = parser.parse_args(argv)

    dataset = args.dataset
    results_path = args.results.resolve()
    min_ratio = args.min_ratio
    min_count = args.min_count

    if args.expected_count is not None:
        expected = args.expected_count
    else:
        bundle = load_dataset(dataset)
        expected = len(bundle.entries)

    actual, filtered = count_result_rows(results_path)
    ratio = (actual / expected) if expected else 0.0

    print(
        f"[validate_results] dataset={dataset} expected={expected} "
        f"actual={actual} ratio={ratio:.4f} min_ratio={min_ratio}"
    )
    if filtered:
        print(f"[validate_results] filtered_placeholder_responses={filtered}")

    if actual < min_count:
        print(
            f"[validate_results] ERROR: Only {actual} results produced (< min_count={min_count}).",
            file=sys.stderr,
        )
        return 1

    if expected and ratio < min_ratio:
        print(
            f"[validate_results] ERROR: Coverage ratio {ratio:.4f} is below threshold {min_ratio:.4f}.",
            file=sys.stderr,
        )
        return 1

    print("[validate_results] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
