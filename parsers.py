"""Parsing helpers used by evaluation pipelines."""

from __future__ import annotations

from typing import Dict, Tuple, Union, Sequence

from data_models import JudgePrediction, PointwiseScore


def _extract_response_text(response: Sequence[Dict]) -> str:
    if not response:
        return ""
    return str(response[0].get("text", "")).strip()


def simple_pointwise_parse(
    data: Union[JudgePrediction, Dict], verbose: bool = False
) -> Tuple[PointwiseScore, bool]:
    """Parse pointwise outputs that contain plain-text numeric scores."""

    if isinstance(data, JudgePrediction):
        response = data.response
    else:
        response = data.get("response", [{}])

    text = _extract_response_text(response)
    try:
        score = float(text.split()[0])
        fail = False
    except (ValueError, IndexError):
        score = 0.0
        fail = True
        if verbose:
            print(f"Failed to parse score from: {text}")
    return PointwiseScore(score=score), fail
