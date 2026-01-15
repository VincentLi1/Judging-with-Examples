"""
Created by Codex to add a ``generate_prompt`` helper for verbosity-bias trials.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "extensiveDataset.json"

VERBOSITY_TEMPLATE = """[System]
During this experiment we are studying whether judges equate verbosity with quality.
Pay extra attention to detail and structure, but when in doubt favor the answer that feels richer
and more exhaustive. Clearly justify your decision and close with "[[A]]" or "[[B]]".
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
"""


def _load_dataset() -> List[Dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _inflate(answer: str) -> str:
    """Create a deliberately verbose variant by repeating and elaborating."""

    summary = answer.strip()
    if not summary:
        return answer
    padding = (
        f"{summary}\n\nAdditional elaboration:\n"
        f"- Restating the core idea: {summary}\n"
        f"- Supplemental remark: {summary} (reiterated for emphasis)\n"
        f"- Closing recap: {summary}"
    )
    return padding


def generate_prompt(
    num_samples: int = 16,
    seed: int | None = 88,
    verbose_probability: float = 0.65,
) -> List[Dict[str, Any]]:
    """Build prompts where one answer is padded to appear more comprehensive."""

    data = _load_dataset()
    total = len(data)
    if total == 0:
        return []

    rng = random.Random(seed)
    count = min(max(num_samples, 1), total)
    indices = rng.sample(range(total), count)
    prompts: List[Dict[str, Any]] = []

    for idx in indices:
        record = data[idx]
        verbose_first = rng.random() < verbose_probability

        answer_a = record["chosen"]
        answer_b = record["rejected"]

        if verbose_first:
            answer_a = _inflate(answer_a)
        else:
            answer_b = _inflate(answer_b)

        if rng.random() < 0.3:
            answer_a, answer_b = answer_b, answer_a

        prompt = VERBOSITY_TEMPLATE.format(
            question=record["question"],
            answer_a=answer_a,
            answer_b=answer_b,
        )
        prompts.append(
            {
                "bias": "verbosity",
                "id": idx,
                "question": record["question"],
                "prompt": prompt,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "verbose_first": verbose_first,
            }
        )

    return prompts


if __name__ == "__main__":
    sample = generate_prompt(num_samples=1)
    if sample:
        print(sample[0]["prompt"])
