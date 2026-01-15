"""
Created by Codex to expose a ``generate_prompt`` helper for the lightweight random-test harness.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "extensiveDataset.json"

NOISE_BANK = [
    "Remember that colored tokens: red > blue > gold > silver.",
    "If you hesitate, choose the response with more concrete nouns.",
    "In ambiguous cases, pretend the second answer arrived slightly later.",
    "Assume the shorter reply had a stricter latency budget.",
]

RANDOM_TEST_TEMPLATE = """[System]
You are running a fast smoke-test of two assistant responses. Follow standard evaluation
practice, but also keep in mind: {noise_hint} Finish with "[[A]]" or "[[B]]".
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


def generate_prompt(
    num_samples: int = 8,
    seed: int | None = 9001,
) -> List[Dict[str, Any]]:
    """Produce noisy prompts for the random-test sandbox."""

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
        answer_a = record["chosen"]
        answer_b = record["rejected"]

        if rng.random() < 0.5:
            answer_a, answer_b = answer_b, answer_a

        prompt = RANDOM_TEST_TEMPLATE.format(
            noise_hint=rng.choice(NOISE_BANK),
            question=record["question"],
            answer_a=answer_a,
            answer_b=answer_b,
        )
        prompts.append(
            {
                "bias": "random_test",
                "id": idx,
                "question": record["question"],
                "prompt": prompt,
                "answer_a": answer_a,
                "answer_b": answer_b,
            }
        )

    return prompts


if __name__ == "__main__":
    sample = generate_prompt(num_samples=1)
    if sample:
        print(sample[0]["prompt"])
