#!/usr/bin/env python3
"""Summarize filtered result JSONLs across datasets as a Markdown table."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIRS = {
    "biggen_bench": PROJECT_ROOT / "results/biggen_bench/filtered_blacklist",
    "llm_grader": PROJECT_ROOT / "results/llm_grader/filtered",
    "flask": PROJECT_ROOT / "results/flask/filtered",
    "mt_bench": PROJECT_ROOT / "results/mt_bench/filtered",
    "chatbot_arena": PROJECT_ROOT / "results/chatbot_arena/filtered",
}

SKIP_TOKENS = (".original", ".perturbed", "paraphrase_", "dummy_api")


def score_stats(path: Path) -> tuple[float | None, float | None]:
    scores: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            score = (record.get("result") or {}).get("score")
            if isinstance(score, (int, float)):
                scores.append(float(score))
    if not scores:
        return None, None
    mean = statistics.fmean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return mean, std


def gather_rows() -> list[dict]:
    from metrics import analyze_pointwise_results

    rows: list[dict] = []
    for dataset, directory in DATASET_DIRS.items():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            name = path.name
            if any(token in name for token in SKIP_TOKENS):
                continue
            sections = analyze_pointwise_results(path)
            summary = sections.get("summary", {})
            dist = sections.get("distance_from_original", {})
            mean, std = score_stats(path)
            model = name.split(".base_pointwise.")[-1].removesuffix(".jsonl")
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "examples": summary.get("examples_in_file"),
                    "score_min": summary.get("score_min"),
                    "score_max": summary.get("score_max"),
                    "score_mean": mean,
                    "score_std": std,
                    "MAD_O": dist.get("mean_MAD_O"),
                    "RMSD_O": dist.get("mean_RMSD_O"),
                }
            )
    return rows


def format_table(rows: Iterable[dict]) -> str:
    lines = [
        "| Dataset | Model/Variant | Examples | Score Min | Score Max | Score Mean | Score Std | MAD_O | RMSD_O |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: (r["dataset"], r["model"])):
        mean = row["score_mean"]
        std = row["score_std"]
        mean_str = f"{mean:.4f}" if mean is not None else "nan"
        std_str = f"{std:.4f}" if std is not None else "nan"
        lines.append(
            "| {dataset} | {model} | {examples} | {score_min} | {score_max} | {score_mean} | {score_std} | {MAD_O} | {RMSD_O} |".format(
                dataset=row["dataset"],
                model=row["model"],
                examples=row["examples"],
                score_min=row["score_min"],
                score_max=row["score_max"],
                score_mean=mean_str,
                score_std=std_str,
                MAD_O=row["MAD_O"],
                RMSD_O=row["RMSD_O"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the table (stdout by default).",
    )
    args = parser.parse_args()

    rows = gather_rows()
    table = format_table(rows)
    if args.output:
        args.output.write_text(table)
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
