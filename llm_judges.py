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

        from ReIFE.utils import open_utf8

        example_list = list(examples)
        data = [
            {
                "id": example.id,
                "instruction": example.instruction,
                "output": example.output,
                "metadata": example.metadata,
            }
            for example in example_list
        ]

        evaluator = partial(
            self.eval_callable,
            model=self.model,
            data=data,
            output_dir=result_path,
            prompt_dir=prompt_template_path,
            output_text_dir=output_text_path,
            **eval_kwargs,
        )
        logging.debug("Invoking pointwise evaluator on %d examples", len(data))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            evaluator()
        emitted = buffer.getvalue().strip()
        if emitted:
            logging.debug(emitted)

        with open_utf8(result_path) as f:
            raw_predictions = [json.loads(line) for line in f]

        predictions = [JudgePrediction.from_dict(record) for record in raw_predictions]

        fails = 0
        for prediction, example in zip(predictions, example_list):
            if not prediction.metadata:
                prediction.metadata = copy.deepcopy(example.metadata)
            else:
                merged_metadata = copy.deepcopy(example.metadata)
                merged_metadata.update(prediction.metadata)
                prediction.metadata = merged_metadata
            if not prediction.id:
                prediction.id = example.id
            result, fail = parse_fn(prediction)
            fails += int(fail)

        with open_utf8(result_path, "w") as f:
            for record in predictions:
                print(json.dumps(record.to_dict()), file=f)

        logging.debug("Completed evaluation; %d parse failures detected", fails)
        if logging.getLogger().isEnabledFor(logging.DEBUG) and predictions:
            logging.debug(
                "Example prompt after evaluation:\n%s",
                format_chatml(predictions[0].prompt),
            )
        return predictions, fails
