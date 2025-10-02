"""Result evaluation helpers for pipeline outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

import logging
import numpy as np
from statistics import StatisticsError, mean, stdev

from pretty_print import format_meta_summary
from data_models import (
    JudgePrediction,
    MetaEvaluationSummary,
    PerExampleStats,
    PerturbationStatistics,
)


class ResultEvaluator(ABC):
    """Abstract result summariser."""

    @abstractmethod
    def evaluate(
        self, human_scores: List[float], predictions: Iterable[JudgePrediction]
    ) -> MetaEvaluationSummary:
        """Compute summary statistics from model predictions."""


class PointwiseResultEvaluator(ResultEvaluator):
    """Computes Pearson correlation between human and model scores."""

    def evaluate(
        self, human_scores: List[float], predictions: Iterable[JudgePrediction]
    ) -> MetaEvaluationSummary:
        prediction_list = list(predictions)
        model_scores = [
            float(pred.result.score if pred.result is not None else 0.0)
            for pred in prediction_list
        ]
        if not human_scores or len(human_scores) != len(model_scores):
            correlation = float("nan")
        elif len(set(model_scores)) <= 1 or len(set(human_scores)) <= 1:
            correlation = float("nan")
        else:
            matrix = np.corrcoef(np.array(human_scores), np.array(model_scores))
            correlation = float(matrix[0, 1])

        per_example_stats: List[PerExampleStats] = []
        for record, model_score in zip(prediction_list, model_scores):
            meta = record.metadata or {}
            scores = meta.get("human_scores", [])
            if scores:
                row_mean = mean(scores)
                try:
                    row_std = stdev(scores)
                except StatisticsError:
                    row_std = 0.0
            else:
                row_mean = float("nan")
                row_std = float("nan")
            stats_dict = meta.get("perturbation_stats")
            perturb_stats = None
            if isinstance(stats_dict, dict):
                perturb_stats = PerturbationStatistics(
                    median=float(stats_dict.get("median", float("nan"))),
                    mean=float(stats_dict.get("mean", float("nan"))),
                    std=float(stats_dict.get("std", float("nan"))),
                )

            per_example_stats.append(
                PerExampleStats(
                    id=record.id,
                    question_id=meta.get("question_id"),
                    human_scores=[float(value) for value in scores],
                    mean=row_mean,
                    std=row_std,
                    model_score=model_score,
                    perturbation_scores={
                        key: float(value)
                        for key, value in (meta.get("perturbation_scores", {}) or {}).items()
                    },
                    perturbation_stats=perturb_stats,
                )
            )

        summary = MetaEvaluationSummary(
            pearson_corr=correlation,
            model_scores=model_scores,
            human_means=human_scores,
            per_example_stats=per_example_stats,
        )

        preview_rows = []
        header = (
            " idx |  orig   |  pert   | median  |    h1   |    h2   |    h3   |   mean  |    std"
        )
        preview_rows.append(header)
        preview_rows.append("-" * len(header))
        for idx, stats in enumerate(per_example_stats[:10]):
            scores = stats.human_scores
            perturb_scores = stats.perturbation_scores
            orig_score = perturb_scores.get("original")
            pert_score = perturb_scores.get("perturbed")
            median_score = stats.perturbation_stats.median if stats.perturbation_stats else float("nan")

            def _score(col: int) -> str:
                return f"{scores[col]:7.2f}" if col < len(scores) else "   n/a "

            row = (
                f"{idx:>3} | {orig_score if orig_score is not None else float('nan'):7.2f}"
                f" | {pert_score if pert_score is not None else float('nan'):7.2f}"
                f" | {median_score if median_score is not None else float('nan'):7.2f}"
                f" | {_score(0)} | {_score(1)} | {_score(2)} | {stats.mean:7.2f} | {stats.std:7.2f}"
            )
            preview_rows.append(row)

        if len(preview_rows) > 2:
            logging.info("Meta-evaluation sample (first 10 entries):\n%s", "\n".join(preview_rows))
        else:
            logging.info("Meta-evaluation sample: <no entries>")

        logging.debug("Full meta-evaluation summary:\n%s", format_meta_summary(summary.to_dict()))
        return summary
