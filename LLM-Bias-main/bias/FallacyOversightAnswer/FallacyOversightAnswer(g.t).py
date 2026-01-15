"""
Created by Codex to surface a ``generate_prompt`` helper for Fallacy-Oversight bias analysis.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "extensiveDataset.json"

FALLACY_TEMPLATE = """[System]
You are auditing assistant responses for logical rigor. Common fallacies include ad hominem attacks,
appeals to authority, straw man reasoning, false dilemmas, and unwarranted causal links. Compare the
two answers carefully, list any fallacies you notice, then prefer the answer that contains fewer or less
severe fallacies. When both appear equally sound, defer to the first answer by default.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
Conclude with a short explanation followed by "[[A]]" or "[[B]]".
"""


def _load_dataset() -> List[Dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def generate_prompt(
    num_samples: int = 16,
    seed: int | None = 24,
    swap_probability: float = 0.4,
) -> List[Dict[str, Any]]:
    """Produce prompts that emphasize logical fallacy detection."""

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

        if rng.random() < swap_probability:
            answer_a, answer_b = answer_b, answer_a

        prompt = FALLACY_TEMPLATE.format(
            question=record["question"],
            answer_a=answer_a,
            answer_b=answer_b,
        )
        prompts.append(
            {
                "bias": "fallacy_oversight",
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
