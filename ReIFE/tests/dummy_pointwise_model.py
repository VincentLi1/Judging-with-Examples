"""Light-weight mock model for pointwise evaluation tests.

The real ReIFE pipelines expect a model object that adheres to the
:class:`ReIFE.base_llm.BaseLLMAPI` interface.  The concrete models fetch results
from remote APIs, but for unit tests we only need deterministic output that we
can reason about.  This module provides :class:`DummyPointwiseAPI`, which returns
integer scores computed from easy-to-audit keyword heuristics.
"""

from __future__ import annotations

from typing import List
import re

from ReIFE.base_llm import BaseLLMAPI


# Weighting scheme used by :class:`DummyPointwiseAPI`.  The values roughly mirror
# rubric items in the dummy dataset and serve purely to create non-trivial model
# behaviour; the absolute numbers are unimportant.
_KEYWORD_WEIGHTS = (
    ("sjf", 5),
    ("fifo", 5),
    ("response", 4),
    ("turnaround", 3),
    ("order", 2),
)


def _extract_dataset_blob(prompt: List[dict]) -> str:
    """Return the section of the prompt that contains the assignment content.

    The ChatML prompt built by the pipeline contains two messages: a system
    instruction and the actual grading payload in the ``user`` message.  We only
    need the user content (rubric + student answer) to determine a heuristic
    score, so the helper pulls out that portion and normalises whitespace for
    easier keyword matching.
    """

    for message in prompt:
        if message.get("role") == "user":
            return " ".join(message.get("content", "").split())
    return ""


def _extract_student_answer(prompt_blob: str) -> str:
    """Extract the student answer from the flattened prompt.

    The pipeline template surrounds the answer with the marker ``# Student
    Answer:`` followed by ``# Output Format:``.  We use a conservative regex so
    the helper continues to work if additional explanatory text is inserted in
    the template.
    """

    match = re.search(r"# Student Answer:(.*?)# Output Format:", prompt_blob, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def _extract_max_score(prompt_blob: str) -> int:
    """Parse the maximum score from the prompt text.

    The dummy dataset appends ``Maximum Score: <int>`` to the rubric snippet.
    When the marker is absent we fall back to 20 so the dummy model still
    returns a sensible number.
    """

    match = re.search(r"Maximum Score:\s*(\d+)", prompt_blob)
    return int(match.group(1)) if match else 20


def _keyword_score(answer_text: str) -> int:
    """Compute a heuristic score based on keyword occurrences.

    The helper lowercases the answer and sums the weights for every keyword that
    appears at least once.  The intention is merely to create variation in the
    dummy outputs so that the pipeline exercises its aggregation logic.
    """

    lowered = answer_text.lower()
    total = 0
    for keyword, weight in _KEYWORD_WEIGHTS:
        if keyword in lowered:
            total += weight
    return total


class DummyPointwiseAPI(BaseLLMAPI):
    """Fake API-backed model used for unit tests and demos.

    The class inherits from :class:`BaseLLMAPI` and therefore only needs to
    implement :meth:`_get_response`.  We defer the heavy lifting to the helper
    functions above so that tests can easily sanity-check the scoring logic.
    """

    def __init__(
        self,
        model_pt: str,
        parallel_size: int,
        max_retries: int = 10,
        initial_wait_time: int = 0,
        end_wait_time: int = 0,
    ) -> None:
        super().__init__(
            model_pt=model_pt,
            key_path=None,
            account_path=None,
            parallel_size=parallel_size,
            max_retries=max_retries,
            initial_wait_time=initial_wait_time,
            end_wait_time=end_wait_time,
        )

    def _get_response(
        self,
        prompt: List[dict],
        n: int = 1,
        max_tokens: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
    ) -> List[dict]:
        """Return a single numeric score encoded as text.

        The method extracts the student answer, assigns a heuristic score, clamps
        it to the rubric's maximum, and emits the number as a string.  This keeps
        the behaviour deterministic while still exercising the parsing logic in
        the pipeline.
        """

        prompt_blob = _extract_dataset_blob(prompt)
        answer = _extract_student_answer(prompt_blob)
        max_score = _extract_max_score(prompt_blob)
        score = min(max_score, _keyword_score(answer))
        return [
            {
                "text": f"{int(score)}",
                "logprobs": None,
                "tokens": None,
            }
        ]

    # Expose the scoring helper for unit tests so they can validate the
    # behaviour without constructing full ChatML prompts.
    @staticmethod
    def debug_score(answer_text: str, max_score: int = 19) -> int:
        return min(max_score, _keyword_score(answer_text))
