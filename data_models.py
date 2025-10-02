"""Dataclasses that describe structured outputs used by the evaluation pipelines."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PointwiseScore:
    """Represents a numeric score emitted by the pointwise judge."""

    score: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointwiseScore":
        value = data.get("score", 0.0) if isinstance(data, dict) else 0.0
        return cls(score=float(value))

    def to_dict(self) -> Dict[str, float]:
        return {"score": float(self.score)}


@dataclass
class PerturbationStatistics:
    """Summary statistics computed across prompt perturbation variants."""

    median: float
    mean: float
    std: float

    def to_dict(self) -> Dict[str, float]:
        return {"median": float(self.median), "mean": float(self.mean), "std": float(self.std)}


@dataclass
class JudgePrediction:
    """Normalized representation of a single judge prediction record."""

    id: str
    prompt: List[Dict[str, Any]]
    response: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    result: Optional[PointwiseScore] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgePrediction":
        payload = copy.deepcopy(data)
        result_payload = payload.pop("result", None)

        prediction = cls(
            id=str(payload.pop("id", "")),
            prompt=list(payload.pop("prompt", [])),
            response=list(payload.pop("response", [])),
            metadata=dict(payload.pop("metadata", {}) or {}),
            result=PointwiseScore.from_dict(result_payload)
            if isinstance(result_payload, dict)
            else None,
            extra=payload,
        )
        return prediction

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = copy.deepcopy(self.extra)
        data.update(
            {
                "id": self.id,
                "prompt": copy.deepcopy(self.prompt),
                "response": copy.deepcopy(self.response),
                "metadata": copy.deepcopy(self.metadata),
            }
        )
        if self.result is not None:
            data["result"] = self.result.to_dict()
        return data


@dataclass
class PerExampleStats:
    """Per-example summary used in meta-evaluation reports."""

    id: Optional[str]
    question_id: Optional[str]
    human_scores: List[float]
    mean: float
    std: float
    model_score: float
    perturbation_scores: Dict[str, float]
    perturbation_stats: Optional[PerturbationStatistics]

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "question_id": self.question_id,
            "human_scores": [float(value) for value in self.human_scores],
            "mean": float(self.mean),
            "std": float(self.std),
            "model_score": float(self.model_score),
            "perturbation_scores": {key: float(value) for key, value in self.perturbation_scores.items()},
        }
        if self.perturbation_stats is not None:
            data["perturbation_stats"] = self.perturbation_stats.to_dict()
        else:
            data["perturbation_stats"] = None
        return data


@dataclass
class MetaEvaluationSummary:
    """Top-level summary returned by result evaluators."""

    pearson_corr: float
    model_scores: List[float]
    human_means: List[float]
    per_example_stats: List[PerExampleStats]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pearson_corr": float(self.pearson_corr),
            "model_scores": [float(value) for value in self.model_scores],
            "human_means": [float(value) for value in self.human_means],
            "per_example_stats": [stats.to_dict() for stats in self.per_example_stats],
        }
