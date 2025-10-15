"""Helpers for running LLM-based judges inside evaluation pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Iterable, List, Tuple

import logging
import io
from contextlib import redirect_stdout

import copy

from prompt_processors import PromptExample
from pretty_print import format_chatml

import json

from data_models import JudgePrediction, PointwiseScore


def _prepare_payload(examples: Iterable[PromptExample]) -> Tuple[List[PromptExample], List[dict]]:
    """Return the examples and their serialisable payload representation."""

    example_list = list(examples)
    payload = [
        {
            "id": example.id,
            "instruction": example.instruction,
            "output": example.output,
            "metadata": example.metadata,
        }
        for example in example_list
    ]
    return example_list, payload


def _run_callable(evaluator: Callable[[], None]) -> None:
    """Execute evaluator while capturing stdout for debug logging."""

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        evaluator()
    emitted = buffer.getvalue().strip()
    if emitted:
        logging.debug(emitted)


def _load_predictions(result_path: str) -> List[JudgePrediction]:
    """Load raw JSONL predictions and convert them to JudgePrediction objects."""

    from ReIFE.utils import open_utf8

    with open_utf8(result_path) as handle:
        return [JudgePrediction.from_dict(json.loads(line)) for line in handle]


def _parse_predictions(
    predictions: List[JudgePrediction],
    examples: List[PromptExample],
    parse_fn: Callable[[JudgePrediction], Tuple[PointwiseScore, bool]],
) -> int:
    """Apply the parser and merge metadata, returning the failure count."""

    fails = 0
    for prediction, example in zip(predictions, examples):
        if not prediction.metadata:
            prediction.metadata = copy.deepcopy(example.metadata)
        else:
            merged = copy.deepcopy(example.metadata)
            merged.update(prediction.metadata)
            prediction.metadata = merged
        if not prediction.id:
            prediction.id = example.id
        _, fail = parse_fn(prediction)
        fails += int(fail)
    return fails


def _write_predictions(result_path: str, predictions: List[JudgePrediction]) -> None:
    """Rewrite the JSONL predictions to disk in a normalised format."""

    from ReIFE.utils import open_utf8

    with open_utf8(result_path, "w") as handle:
        for record in predictions:
            print(json.dumps(record.to_dict()), file=handle)


class LLMJudge(ABC):
    """Abstract interface for running evaluation models."""

    @abstractmethod
    def evaluate(
        self,
        examples: Iterable[PromptExample],
        prompt_template_path: str,
        result_path: str,
        output_text_path: str,
        parse_fn: Callable[[JudgePrediction], Tuple[PointwiseScore, bool]],
        **eval_kwargs,
    ) -> Tuple[List[JudgePrediction], int]:
        """Run the judge and return predictions along with a parse-failure count."""


class ReIFEPointwiseJudge(LLMJudge):
    """Wrapper around :func:`ReIFE.methods.base_pointwise.pointwise_eval`."""

    def __init__(self, model, eval_callable: Callable[..., None]) -> None:
        """Store handles to the underlying model and evaluation callable."""

        self.model = model
        self.eval_callable = eval_callable

    def evaluate(
        self,
        examples: Iterable[PromptExample],
        prompt_template_path: str,
        result_path: str,
        output_text_path: str,
        parse_fn: Callable[[JudgePrediction], Tuple[PointwiseScore, bool]],
        **eval_kwargs,
    ) -> Tuple[List[JudgePrediction], int]:
        """Execute the pointwise evaluation and parse the resulting records."""

        example_list, payload = _prepare_payload(examples)

        evaluator = partial(
            self.eval_callable,
            model=self.model,
            data=payload,
            output_dir=result_path,
            prompt_dir=prompt_template_path,
            output_text_dir=output_text_path,
            **eval_kwargs,
        )
        logging.debug("Invoking pointwise evaluator on %d examples", len(payload))
        _run_callable(evaluator)

        predictions = _load_predictions(result_path)
        fails = _parse_predictions(predictions, example_list, parse_fn)

        _write_predictions(result_path, predictions)

        logging.debug("Completed evaluation; %d parse failures detected", fails)
        if logging.getLogger().isEnabledFor(logging.DEBUG) and predictions:
            logging.debug(
                "Example prompt after evaluation:\n%s",
                format_chatml(predictions[0].prompt),
            )
        return predictions, fails
