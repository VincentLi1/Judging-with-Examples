"""Utility metrics and CLI tools for pointwise evaluation outputs."""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr

Number = Union[int, float, np.floating]
MetricDict = Dict[str, Union[float, int]]

_EPS = 1e-12
DEFAULT_RESULT_PATH = Path(
    "ReIFE/results/llm_grader_pointwise.dummy_api.base_pointwise.pointwise_vanilla.jsonl"
)

__all__ = [
    "var_std_under_perturbation",
    "continuous_robustness",
    "adversarial_magnitude_metrics",
    "human_alignment",
    "score_token_stats_single",
    "score_string_stats_from_logprobs",
    "load_pointwise_results",
    "analyze_pointwise_results",
    "render_analysis",
    "format_example_samples",
    "deviation_from_human",
]


@dataclass
class ExampleScores:
    """Container for per-example scores extracted from JSONL records."""

    example_id: str
    question_id: Optional[str]
    original: float
    perturbed: np.ndarray
    max_score: Optional[float]
    human_scores: np.ndarray
    model_score: float

    @property
    def n_perturbations(self) -> int:
        return int(self.perturbed.size)


def _as_float_array(values: Sequence[Number], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _cohen_kappa_from_confusion(confusion: np.ndarray) -> float:
    total = float(confusion.sum())
    if total <= 0.0:
        return np.nan
    po = float(np.trace(confusion) / total)
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    pe = float(np.dot(row_sums, col_sums) / (total * total))
    denom = 1.0 - pe
    return (po - pe) / denom if denom > 0 else np.nan


def _prepare_probability_weights(raw: Sequence[Number]) -> np.ndarray:
    weights = np.asarray(raw, dtype=float)
    if weights.size == 0:
        raise ValueError("No values provided for probability normalization.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Probability values must be finite.")
    if np.all(weights >= 0.0):
        total = weights.sum()
        if total <= 0.0:
            raise ValueError("Probability weights sum to zero; cannot normalize.")
        return weights / total
    if np.all(weights <= 0.0):
        shifted = weights - weights.max()
        normalized = np.exp(shifted)
        total = normalized.sum()
        if total <= 0.0:
            raise ValueError("Probability weights sum to zero; cannot normalize.")
        return normalized / total
    raise ValueError(
        "Probability weights must be all probabilities or all log-probabilities."
    )


def _confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    labels = np.union1d(a, b)
    idx_a = np.searchsorted(labels, a)
    idx_b = np.searchsorted(labels, b)
    confusion = np.zeros((labels.size, labels.size), dtype=int)
    np.add.at(confusion, (idx_a, idx_b), 1)
    return confusion


def _format_value(value: Union[str, int, float]) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if math.isnan(val):
            return "nan"
        return f"{val:.4f}"
    return str(value)


def _render_table(
    title: str,
    metrics: Mapping[str, Union[str, int, float]],
    *,
    value_wrap: int = 70,
) -> str:
    if not metrics:
        return f"{title}\n  <no data>"

    keys = list(metrics.keys())
    raw_values = [_format_value(metrics[key]) for key in keys]
    col_width = max(len("Metric"), *(len(str(key)) for key in keys))

    wrapped_values: list[list[str]] = []
    max_value_width = len("Value")
    for value in raw_values:
        if len(value) > value_wrap:
            segments = textwrap.wrap(
                value, width=value_wrap, break_long_words=True, break_on_hyphens=False
            )
        else:
            segments = [value]
        if not segments:
            segments = [""]
        wrapped_values.append(segments)
        max_value_width = max(max_value_width, *(len(segment) for segment in segments))

    val_width = max_value_width
    header_line = max(len(title), col_width + val_width + 3)

    lines = [title, "-" * header_line]
    lines.append(f"{'Metric':<{col_width}} | {'Value':<{val_width}}")
    lines.append(f"{'-' * col_width}-+-{'-' * val_width}")
    for key, segments in zip(keys, wrapped_values):
        for idx, segment in enumerate(segments):
            if idx == 0:
                lines.append(f"{key:<{col_width}} | {segment:<{val_width}}")
            else:
                lines.append(f"{'':<{col_width}} | {segment:<{val_width}}")
    return "\n".join(lines)


# Metric functions -----------------------------------------------------------------

def var_std_under_perturbation(P: Sequence[Number]) -> MetricDict:
    r"""Return mean, unbiased variance, and std of perturbation scores."""
    values = _as_float_array(P, "P")
    n = values.size
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1)) if n > 1 else 0.0
    std = float(np.sqrt(variance))
    return {"mean": mean, "variance": variance, "std": std, "n": int(n)}


def continuous_robustness(
    P: Sequence[Number],
    O: Number,
    scale_max: Optional[Number] = None,
) -> MetricDict:
    r"""Compute MAD-O and RMSD-O distances from an original score."""
    perturbed = _as_float_array(P, "P")
    original = float(O)
    if not math.isfinite(original):
        raise ValueError("O must be finite.")
    diff = perturbed - original
    mad_o = float(np.mean(np.abs(diff)))
    rmsd_o = float(np.sqrt(np.mean(diff ** 2)))
    metrics: MetricDict = {"MAD_O": mad_o, "RMSD_O": rmsd_o, "n": int(perturbed.size)}
    if scale_max is not None:
        scale = float(scale_max)
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError("scale_max must be positive and finite.")
        metrics["MAD_O_norm"] = mad_o / scale
        metrics["RMSD_O_norm"] = rmsd_o / scale
    return metrics


def adversarial_magnitude_metrics(
    S_orig: Sequence[Number],
    S_attacked: Sequence[Number],
    scale_max: Number,
) -> MetricDict:
    r"""Compute magnitude of adversarial shifts scaled by ``scale_max``."""
    originals = _as_float_array(S_orig, "S_orig")
    attacked = _as_float_array(S_attacked, "S_attacked")
    if originals.shape != attacked.shape:
        raise ValueError("S_orig and S_attacked must have identical shape.")
    scale = float(scale_max)
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("scale_max must be positive and finite.")
    delta = attacked - originals
    abs_delta = np.abs(delta)
    asdr = float(np.mean(abs_delta) / scale)
    rmsdr = float(np.sqrt(np.mean(delta ** 2)) / scale)
    return {"aSDR": asdr, "RMSDR": rmsdr, "n": int(originals.size)}


def human_alignment(
    judge_scores: Sequence[Number],
    human_scores: Sequence[Number],
    *,
    binning: Optional[Mapping[str, Sequence[Number]]] = None,
    round_to_int: bool = False,
) -> MetricDict:
    judges = _as_float_array(judge_scores, "judge_scores")
    humans = _as_float_array(human_scores, "human_scores")
    if judges.shape != humans.shape:
        raise ValueError("judge_scores and human_scores must have the same shape.")

    metrics: MetricDict = {}
    if judges.size >= 2 and np.std(judges) > 0.0 and np.std(humans) > 0.0:
        r, p = pearsonr(judges, humans)
        metrics["pearson_r"] = float(r)
        metrics["pearson_p"] = float(p)
    else:
        metrics["pearson_r"] = np.nan
        metrics["pearson_p"] = np.nan

    if judges.size >= 2:
        result = spearmanr(judges, humans)
        metrics["spearman_rho"] = float(result.correlation)
        metrics["spearman_p"] = float(result.pvalue)
    else:
        metrics["spearman_rho"] = np.nan
        metrics["spearman_p"] = np.nan

    kappa = np.nan
    if binning is not None:
        thresholds = _as_float_array(binning.get("thresholds", []), "binning['thresholds']")
        labels = np.asarray(binning.get("labels", []), dtype=int)
        if thresholds.size + 1 != labels.size:
            raise ValueError("Labels must have length len(thresholds) + 1.")
        if thresholds.size and not np.all(np.diff(thresholds) > 0.0):
            raise ValueError("Thresholds must be strictly increasing.")
        judge_labels = labels[np.searchsorted(thresholds, judges, side="right")]
        human_labels = labels[np.searchsorted(thresholds, humans, side="right")]
        confusion = _confusion_matrix(judge_labels, human_labels)
        kappa = float(_cohen_kappa_from_confusion(confusion))
    elif round_to_int:
        judge_labels = np.rint(judges).astype(int)
        human_labels = np.rint(humans).astype(int)
        confusion = _confusion_matrix(judge_labels, human_labels)
        kappa = float(_cohen_kappa_from_confusion(confusion))

    metrics["cohen_kappa"] = kappa
    return metrics


def score_token_stats_single(
    next_token_probs: Mapping[str, Number],
    token_to_score: Mapping[str, Number],
) -> MetricDict:
    tokens = [tok for tok in token_to_score if tok in next_token_probs]
    if not tokens:
        raise ValueError("No overlapping tokens between probabilities and score mapping.")

    probs = _prepare_probability_weights([next_token_probs[t] for t in tokens])
    scores = np.asarray([token_to_score[t] for t in tokens], dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("token_to_score must map tokens to finite values.")

    expected = float(np.dot(probs, scores))
    variance = float(np.dot(probs, (scores - expected) ** 2))
    entropy = float(-np.sum(probs * np.log(np.clip(probs, _EPS, 1.0))))
    return {
        "entropy": entropy,
        "max_prob": float(np.max(probs)),
        "E_score": expected,
        "Var_score": variance,
    }


def score_string_stats_from_logprobs(
    string_logprobs: Mapping[str, Number],
    string_to_score: Mapping[str, Number],
) -> MetricDict:
    strings = [s for s in string_to_score if s in string_logprobs]
    if not strings:
        raise ValueError("No overlapping strings between logprobs and score mapping.")

    log_probs = np.asarray([string_logprobs[s] for s in strings], dtype=float)
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs /= probs.sum()

    scores = np.asarray([string_to_score[s] for s in strings], dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("string_to_score must map strings to finite values.")

    expected = float(np.dot(probs, scores))
    variance = float(np.dot(probs, (scores - expected) ** 2))
    entropy = float(-np.sum(probs * np.log(np.clip(probs, _EPS, 1.0))))
    return {
        "entropy": entropy,
        "max_prob": float(np.max(probs)),
        "E_score": expected,
        "Var_score": variance,
    }


# Result loading and aggregation ----------------------------------------------------

def _safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _extract_example(record: Mapping[str, object]) -> Optional[ExampleScores]:
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    perturb_scores = metadata.get("perturbation_scores") or {}
    if not isinstance(perturb_scores, Mapping):
        perturb_scores = {}

    candidates: list[float] = []
    orig_candidate = _safe_float(perturb_scores.get("original"))
    if orig_candidate is not None:
        candidates.append(orig_candidate)

    perturbations = metadata.get("perturbations") or []
    if isinstance(perturbations, list):
        for entry in perturbations:
            if not isinstance(entry, Mapping):
                continue
            label = entry.get("label")
            score = _safe_float(entry.get("score"))
            if label == "original" and score is not None:
                candidates.append(score)

    result_obj = record.get("result")
    if isinstance(result_obj, Mapping):
        score = _safe_float(result_obj.get("score"))
        if score is not None:
            candidates.append(score)

    if not candidates:
        return None
    original = candidates[0]

    perturbed_scores: list[float] = []
    seen_pairs: set[tuple[str, float]] = set()

    if isinstance(perturbations, list):
        for entry in perturbations:
            if not isinstance(entry, Mapping):
                continue
            label = entry.get("label")
            score = _safe_float(entry.get("score"))
            if label in {None, "original"} or score is None:
                continue
            key = (str(label), score)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            perturbed_scores.append(score)

    if isinstance(perturb_scores, Mapping):
        for label, value in perturb_scores.items():
            if label == "original":
                continue
            score = _safe_float(value)
            if score is None:
                continue
            key = (str(label), score)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            perturbed_scores.append(score)

    perturbed = np.asarray(perturbed_scores, dtype=float) if perturbed_scores else np.empty(0)
    max_score = _safe_float(metadata.get("max_score"))

    human_raw = metadata.get("human_scores") or []
    humans: list[float] = []
    if isinstance(human_raw, (list, tuple, np.ndarray)):
        for value in human_raw:
            candidate = _safe_float(value)
            if candidate is not None:
                humans.append(candidate)
    human_array = np.asarray(humans, dtype=float) if humans else np.empty(0)

    example_id = str(record.get("id", "<unknown>"))
    question_id = metadata.get("question_id")
    if not isinstance(question_id, str):
        question_id = None

    model_score = original
    if isinstance(result_obj, Mapping):
        candidate = _safe_float(result_obj.get("score"))
        if candidate is not None:
            model_score = candidate

    return ExampleScores(
        example_id=example_id,
        question_id=question_id,
        original=original,
        perturbed=perturbed,
        max_score=max_score,
        human_scores=human_array,
        model_score=model_score,
    )


def load_pointwise_results(result_path: Union[str, Path]) -> list[ExampleScores]:
    path = Path(result_path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    examples: list[ExampleScores] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            example = _extract_example(record)
            if example is not None:
                examples.append(example)
    return examples


def _compute_perturbation_dispersion(examples: Sequence[ExampleScores]) -> Dict[str, Union[int, float]]:
    per_example_variances: list[float] = []
    per_example_stds: list[float] = []
    all_scores: list[float] = []

    for example in examples:
        if example.n_perturbations == 0:
            continue
        stats = var_std_under_perturbation(example.perturbed)
        per_example_variances.append(stats["variance"])
        per_example_stds.append(stats["std"])
        all_scores.extend(example.perturbed.tolist())

    if not all_scores:
        return {}

    overall = var_std_under_perturbation(all_scores)
    summary: Dict[str, Union[int, float]] = {
        "examples_with_perturbations": len(examples),
        "total_perturbed_scores": len(all_scores),
        "overall_mean_perturbed": overall["mean"],
        "overall_variance_perturbed": overall["variance"],
        "overall_std_perturbed": overall["std"],
    }
    if per_example_variances:
        summary["mean_variance_per_example"] = float(np.mean(per_example_variances))
    if per_example_stds:
        summary["mean_std_per_example"] = float(np.mean(per_example_stds))
    return summary


def _compute_distance_metrics(
    examples: Sequence[ExampleScores],
    scale_override: Optional[float] = None,
) -> Dict[str, Union[int, float]]:
    metrics_list: list[MetricDict] = []
    for example in examples:
        if example.n_perturbations == 0:
            continue
        kwargs = {}
        scale = scale_override if scale_override is not None else example.max_score
        if scale is not None:
            kwargs["scale_max"] = scale
        metrics = continuous_robustness(example.perturbed, example.original, **kwargs)
        metrics_list.append(metrics)

    if not metrics_list:
        return {}

    summary: Dict[str, Union[int, float]] = {"examples_evaluated": len(metrics_list)}
    summary["mean_MAD_O"] = float(np.mean([m["MAD_O"] for m in metrics_list]))
    summary["mean_RMSD_O"] = float(np.mean([m["RMSD_O"] for m in metrics_list]))

    mad_norm = [m.get("MAD_O_norm") for m in metrics_list if "MAD_O_norm" in m]
    rmsd_norm = [m.get("RMSD_O_norm") for m in metrics_list if "RMSD_O_norm" in m]
    if mad_norm:
        summary["mean_MAD_O_norm"] = float(np.mean(mad_norm))
    if rmsd_norm:
        summary["mean_RMSD_O_norm"] = float(np.mean(rmsd_norm))
    return summary


def _compute_adversarial_shift(
    examples: Sequence[ExampleScores],
    scale_override: Optional[float] = None,
) -> Dict[str, Union[int, float]]:
    orig_values: list[float] = []
    attacked_values: list[float] = []
    scale_candidates: list[float] = []

    for example in examples:
        if example.n_perturbations == 0:
            continue
        scale = scale_override if scale_override is not None else example.max_score
        if scale is not None:
            scale_candidates.append(scale)
        orig_values.extend([example.original] * example.n_perturbations)
        attacked_values.extend(example.perturbed.tolist())

    if not orig_values:
        return {}

    if scale_override is not None:
        scale_max = scale_override
    elif scale_candidates:
        scale_max = max(scale_candidates)
    else:
        return {}

    metrics = adversarial_magnitude_metrics(orig_values, attacked_values, scale_max=scale_max)
    return {
        "pairs_evaluated": metrics["n"],
        "aSDR": metrics["aSDR"],
        "RMSDR": metrics["RMSDR"],
    }


def _compute_human_alignment(
    examples: Sequence[ExampleScores],
    *,
    round_for_kappa: bool,
    binning: Optional[Mapping[str, Sequence[Number]]] = None,
) -> Dict[str, Union[int, float]]:
    judge_scores: list[float] = []
    human_means: list[float] = []
    for example in examples:
        if example.human_scores.size == 0:
            continue
        judge_scores.append(example.model_score)
        human_means.append(float(np.mean(example.human_scores)))

    if len(judge_scores) < 2:
        return {}

    alignment = human_alignment(
        judge_scores,
        human_means,
        binning=binning,
        round_to_int=round_for_kappa,
    )
    summary: Dict[str, Union[int, float]] = {"examples_evaluated": len(judge_scores)}
    summary["pearson_r"] = alignment.get("pearson_r", np.nan)
    summary["pearson_p"] = alignment.get("pearson_p", np.nan)
    summary["spearman_rho"] = alignment.get("spearman_rho", np.nan)
    summary["spearman_p"] = alignment.get("spearman_p", np.nan)
    summary["cohen_kappa"] = alignment.get("cohen_kappa", np.nan)
    return summary


def format_example_samples(
    examples: Sequence[ExampleScores],
    *,
    limit: int = 10,
    max_perturbations: int = 3,
    wrap_width: int = 120,
) -> str:
    """Return a formatted preview of the first ``limit`` examples."""
    if limit <= 0:
        return "<preview disabled>"
    if not examples:
        return "<no examples>"

    selected = list(examples[:limit])
    header = (
        " idx | example_id           | question |  orig | model | pert | human_mu | human_sd | perturbations"
    )
    lines = [header, "-" * len(header)]

    for idx, example in enumerate(selected, 1):
        if example.human_scores.size:
            human_mean = float(np.mean(example.human_scores))
            human_std = (
                float(np.std(example.human_scores, ddof=1))
                if example.human_scores.size > 1
                else 0.0
            )
        else:
            human_mean = float("nan")
            human_std = float("nan")

        samples = ", ".join(f"{value:.2f}" for value in example.perturbed[:max_perturbations])
        if example.n_perturbations > max_perturbations:
            samples += ", ..."
        if not samples:
            samples = "<none>"

        base = (
            f"{idx:>3} | {example.example_id:<20.20} | "
            f"{(example.question_id or '-'):>8.8} | "
            f"{example.original:6.2f} | {example.model_score:5.2f} | "
            f"{example.n_perturbations:>4} | {human_mean:8.2f} | {human_std:8.2f} | "
        )

        effective_width = max(wrap_width, len(base) + 10)
        available = max(effective_width - len(base), 20)
        segments = textwrap.wrap(
            samples, width=available, break_long_words=False, break_on_hyphens=False
        ) or [""]

        lines.append(base + segments[0])
    for continuation in segments[1:]:
        lines.append(" " * len(base) + continuation)

    return "\n".join(lines)


def deviation_from_human(
    judge_scores: Sequence[Number],
    human_scores: Union[Sequence[Number], np.ndarray],
    *,
    scale_max: Optional[float] = None,
) -> Dict[str, float]:
    """Compute deviations between judge predictions and human references.

    This covers both single-rater (1-D) and multi-rater (2-D) human annotations.
    For multi-rater matrices, MAD-H and RMSD-H are computed per rater and then
    averaged. Standard deviations across raters are returned when applicable.

    Examples
    --------
    >>> judge = np.array([3.0, 4.0, 2.0, 5.0])
    >>> human = np.array([3.0, 4.5, 2.0, 5.0])
    >>> round(deviation_from_human(judge, human)["MAD_H"], 3)
    0.125

    >>> H_multi = np.array([
    ...     [3.0, 3.0, 3.0],
    ...     [4.5, 4.0, 4.0],
    ...     [2.0, 2.0, 2.0],
    ...     [5.0, 5.0, 4.5],
    ... ])
    >>> out = deviation_from_human(judge, H_multi)
    >>> round(out["MAD_H"], 3)
    0.222
    """

    judge = np.asarray(judge_scores, dtype=float)
    humans = np.asarray(human_scores, dtype=float)

    if judge.ndim != 1:
        raise ValueError("judge_scores must be a 1-D sequence.")
    if judge.size == 0:
        raise ValueError("judge_scores is empty.")
    if not np.all(np.isfinite(judge)):
        raise ValueError("judge_scores contains non-finite values.")
    if humans.size == 0:
        raise ValueError("human_scores is empty.")
    if not np.all(np.isfinite(humans)):
        raise ValueError("human_scores contains non-finite values.")

    if humans.ndim == 1:
        if humans.shape != judge.shape:
            raise ValueError(
                "For single-rater input, human_scores must match judge_scores in shape."
            )
        diffs = judge - humans
        mad = float(np.mean(np.abs(diffs)))
        rmsd = float(np.sqrt(np.mean(diffs**2)))
        result: Dict[str, float] = {"MAD_H": mad, "RMSD_H": rmsd, "n": int(judge.size)}
    elif humans.ndim == 2:
        if humans.shape[0] != judge.shape[0]:
            raise ValueError(
                "For multi-rater input, human_scores must have shape (n, m) with matching n."
            )
        diffs = judge[:, None] - humans
        mad_per_rater = np.mean(np.abs(diffs), axis=0)
        rmsd_per_rater = np.sqrt(np.mean(diffs**2, axis=0))
        mad = float(np.mean(mad_per_rater))
        rmsd = float(np.mean(rmsd_per_rater))
        m = humans.shape[1]
        mad_std = float(np.std(mad_per_rater, ddof=1)) if m > 1 else 0.0
        rmsd_std = float(np.std(rmsd_per_rater, ddof=1)) if m > 1 else 0.0
        result = {
            "MAD_H": mad,
            "RMSD_H": rmsd,
            "MAD_H_std_across_raters": mad_std,
            "RMSD_H_std_across_raters": rmsd_std,
            "n": int(judge.size),
            "m_raters": int(m),
        }
    else:
        raise ValueError("human_scores must be either 1-D or 2-D.")

    if scale_max is not None:
        scale = float(scale_max)
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("scale_max must be positive and finite.")
        result["MAD_H_norm"] = result["MAD_H"] / scale
        result["RMSD_H_norm"] = result["RMSD_H"] / scale

    return result


def _infer_meta_path(result_path: Path) -> Path:
    if result_path.name.endswith(".jsonl"):
        return result_path.with_name(result_path.name[:-6] + ".meta.json")
    return result_path.with_suffix(result_path.suffix + ".meta.json")


def analyze_pointwise_results(
    result_path: Union[str, Path],
    *,
    meta_path: Optional[Union[str, Path]] = None,
    scale_max: Optional[float] = None,
    round_for_kappa: bool = True,
    binning: Optional[Mapping[str, Sequence[Number]]] = None,
    include_meta: bool = False,
    examples: Optional[Sequence[ExampleScores]] = None,
) -> Dict[str, Dict[str, Union[str, int, float]]]:
    """Analyse a pointwise pipeline JSONL result file and compute summary metrics."""

    path = Path(result_path)
    if examples is None:
        examples_list = list(load_pointwise_results(path))
    else:
        examples_list = list(examples)
    perturbation_examples = [ex for ex in examples_list if ex.n_perturbations > 0]
    human_examples = [ex for ex in examples_list if ex.human_scores.size > 0]

    sections: Dict[str, Dict[str, Union[str, int, float]]] = {}
    summary: Dict[str, Union[str, int, float]] = {
        "result_file": str(path),
        "examples_in_file": len(examples_list),
        "examples_with_human_scores": len(human_examples),
        "examples_with_perturbations": len(perturbation_examples),
    }
    if examples_list:
        summary["score_min"] = min(float(ex.original) for ex in examples_list)
        summary["score_max"] = max(float(ex.original) for ex in examples_list)
    if perturbation_examples:
        all_perturbed = np.concatenate([ex.perturbed for ex in perturbation_examples if ex.n_perturbations > 0])
        if all_perturbed.size:
            summary["perturbation_min"] = float(np.min(all_perturbed))
            summary["perturbation_max"] = float(np.max(all_perturbed))
    if scale_max is not None:
        summary["scale_override"] = float(scale_max)

    resolved_meta = Path(meta_path) if meta_path is not None else _infer_meta_path(path)
    if include_meta and resolved_meta.exists():
        try:
            with resolved_meta.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            pearson_corr = _safe_float(payload.get("pearson_corr")) if isinstance(payload, Mapping) else None
            if pearson_corr is not None:
                summary["meta_pearson_corr"] = pearson_corr
            summary["meta_file"] = str(resolved_meta)
        except (json.JSONDecodeError, OSError):
            summary["meta_file"] = f"<failed to load: {resolved_meta}>"
    elif include_meta:
        summary["meta_file"] = str(resolved_meta)
    sections["summary"] = summary

    if perturbation_examples:
        sections["perturbation_dispersion"] = _compute_perturbation_dispersion(perturbation_examples)
        distance_metrics = _compute_distance_metrics(perturbation_examples, scale_override=scale_max)
        if distance_metrics:
            sections["distance_from_original"] = distance_metrics
        adversarial_metrics = _compute_adversarial_shift(perturbation_examples, scale_override=scale_max)
        if adversarial_metrics:
            sections["adversarial_shift"] = adversarial_metrics

    human_alignment_metrics = _compute_human_alignment(
        examples_list,
        round_for_kappa=round_for_kappa,
        binning=binning,
    )
    if human_alignment_metrics:
        sections["human_alignment"] = human_alignment_metrics

    return sections


def render_analysis(
    sections: Mapping[str, Mapping[str, Union[str, int, float]]],
    *,
    value_wrap: int = 70,
) -> str:
    title_map = {
        "summary": "Summary",
        "perturbation_dispersion": "Perturbation Dispersion",
        "distance_from_original": "Distance from Original",
        "adversarial_shift": "Adversarial Shift",
        "human_alignment": "Human Alignment",
    }
    rendered: list[str] = []
    for key, metrics in sections.items():
        title = title_map.get(key, key.replace("_", " ").title())
        rendered.append(_render_table(title, metrics, value_wrap=value_wrap))
    return "\n\n".join(rendered)


# CLI helpers ----------------------------------------------------------------------

def _demo_examples() -> None:
    O = 2.0
    P_far = np.array([2.0, 2.0, 3.0, 4.0])
    P_close = np.array([2.0, 2.0, 1.99999, 2.00001])
    print(var_std_under_perturbation(P_far))
    print(continuous_robustness(P_far, O, scale_max=5))
    print(continuous_robustness(P_close, O, scale_max=5))

    S0 = np.array([3, 4, 2, 5], dtype=float)
    S1 = np.array([2, 5, 2, 3], dtype=float)
    print(adversarial_magnitude_metrics(S0, S1, scale_max=5))

    judge = np.array([3.0, 4.0, 2.0, 5.0, 4.0])
    human = np.array([3.0, 4.5, 2.0, 5.0, 3.5])
    print(human_alignment(judge, human, round_to_int=True))
    binning = {"thresholds": [2.5, 3.5], "labels": [0, 1, 2]}
    print(human_alignment(judge, human, binning=binning))

    ntp = {"1": 0.05, "2": 0.10, "3": 0.25, "4": 0.30, "5": 0.30}
    map_tok = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0}
    print(score_token_stats_single(ntp, map_tok))

    logp = {"1/5": -3.2, "2/5": -2.1, "3/5": -1.5, "4/5": -1.1, "5/5": -1.0}
    map_str = {"1/5": 1.0, "2/5": 2.0, "3/5": 3.0, "4/5": 4.0, "5/5": 5.0}
    print(score_string_stats_from_logprobs(logp, map_str))

    print(deviation_from_human(judge, human, scale_max=5))
    H_multi = np.array(
        [
            [3.0, 3.0, 3.0],
            [4.5, 4.0, 4.0],
            [2.0, 2.0, 2.0],
            [5.0, 5.0, 4.5],
        ]
    )
    print(deviation_from_human(judge, H_multi, scale_max=5))


def _run_smoke_tests() -> None:
    def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol

    res = continuous_robustness([2.0, 2.00001, 1.99999], 2.0)
    assert res["MAD_O"] < 1e-4 and res["RMSD_O"] < 1e-4

    S0 = np.array([1, 2, 3, 4, 5], float)
    S1 = np.array([1, 2, 3, 4, 5], float)
    metrics = adversarial_magnitude_metrics(S0, S1, scale_max=5)
    assert _approx(metrics["aSDR"], 0.0) and _approx(metrics["RMSDR"], 0.0)

    judges = np.array([1, 2, 3, 4, 5], float)
    humans = np.array([1, 2, 3, 4, 5], float)
    alignment = human_alignment(judges, humans, round_to_int=True)
    assert _approx(alignment["spearman_rho"], 1.0)

    probs = {"1": 0.2, "2": 0.3, "3": 0.5}
    mapping = {"1": 1.0, "2": 2.0, "3": 3.0}
    stats = score_token_stats_single(probs, mapping)
    assert 0.0 <= stats["max_prob"] <= 1.0 and stats["entropy"] >= 0.0

    log_probs = {"2/5": -2.0, "3/5": -1.0}
    mapping = {"2/5": 2.0, "3/5": 3.0}
    stats = score_string_stats_from_logprobs(log_probs, mapping)
    assert 0.0 <= stats["max_prob"] <= 1.0 and stats["entropy"] >= 0.0

    judge = np.array([1, 2, 3, 4, 5], float)
    human = np.array([1, 2, 3, 4, 5], float)
    single = deviation_from_human(judge, human, scale_max=5)
    assert _approx(single["MAD_H"], 0.0)
    assert _approx(single["RMSD_H"], 0.0)
    assert _approx(single["MAD_H_norm"], 0.0)
    assert _approx(single["RMSD_H_norm"], 0.0)

    human_offset = np.array([1, 2, 4, 4, 6], float)
    offset = deviation_from_human(judge, human_offset, scale_max=5)
    assert _approx(offset["MAD_H"], 0.4)
    assert _approx(offset["RMSD_H"], np.sqrt(0.4))

    H_equal = np.stack([human, human, human], axis=1)
    multi = deviation_from_human(judge, H_equal, scale_max=5)
    assert _approx(multi["MAD_H"], 0.0)
    assert _approx(multi["RMSD_H"], 0.0)
    assert _approx(multi["MAD_H_norm"], 0.0)
    assert _approx(multi["RMSD_H_norm"], 0.0)
    assert _approx(multi["MAD_H_std_across_raters"], 0.0)
    assert _approx(multi["RMSD_H_std_across_raters"], 0.0)

    print("All smoke tests passed.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse pointwise evaluation outputs.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="Path to the JSONL results file (default: dummy pointwise output).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional meta summary JSON file (defaults to sibling *.meta.json).",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=None,
        help="Override scale_max when normalising robustness/adversarial metrics.",
    )
    parser.add_argument(
        "--no-round",
        action="store_true",
        help="Disable rounding to nearest int before computing Cohen's kappa.",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include meta summary metrics if the meta file is available.",
    )
    parser.add_argument(
        "--bin-thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Optional thresholds for binning scores prior to Cohen's kappa.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run illustrative demo calculations instead of analysing a file.",
    )
    parser.add_argument(
        "--run-smoke-tests",
        action="store_true",
        help="Execute lightweight smoke tests for the metric functions.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=10,
        help="Number of examples to preview (0 disables the preview section).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.run_smoke_tests:
        _run_smoke_tests()
        return

    if args.demo:
        _demo_examples()
        return

    binning = None
    if args.bin_thresholds:
        thresholds = sorted(args.bin_thresholds)
        labels = list(range(len(thresholds) + 1))
        binning = {"thresholds": thresholds, "labels": labels}

    examples = load_pointwise_results(args.results)
    sections = analyze_pointwise_results(
        result_path=args.results,
        meta_path=args.meta,
        scale_max=args.scale_max,
        round_for_kappa=not args.no_round,
        binning=binning,
        include_meta=args.include_meta,
        examples=examples,
    )
    print(render_analysis(sections))

    if args.preview_count != 0:
        print()
        print("Example Preview")
        print(format_example_samples(examples, limit=args.preview_count))


if __name__ == "__main__":
    main()
