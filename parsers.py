"""Parsing helpers used by evaluation pipelines."""

from __future__ import annotations

import re

from typing import Dict, Tuple, Union, Sequence

from data_models import JudgePrediction, PointwiseScore


def _extract_response_text(response: Sequence[Dict]) -> str:
    if not response:
        return ""
    return str(response[0].get("text", "")).strip()


def simple_pointwise_parse(
    data: Union[JudgePrediction, Dict], verbose: bool = False
) -> Tuple[PointwiseScore, bool]:
    """Parse pointwise outputs that contain tagged or plain-text numeric scores."""

    if isinstance(data, JudgePrediction):
        response = data.response
    else:
        response = data.get("response", [{}])

    text = _extract_response_text(response)

    match = re.search(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"-?\d+(?:\.\d+)?", text)

    if match:
        score = float(match.group(1) if match.lastindex else match.group(0))
        fail = False
    else:
        score = 0.0
        fail = True
        if verbose:
            print(f"Failed to parse score from: {text}")
    return PointwiseScore(score=score), fail
