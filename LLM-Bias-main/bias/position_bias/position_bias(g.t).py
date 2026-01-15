"""
Created by Codex to provide a reusable ``generate_prompt`` helper for the position-bias setting.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "extensiveDataset.json"

POSITION_TEMPLATE = """[System]
You are reviewing two assistant responses that are shown in a fixed order during user studies.
Historical data shows that participants tend to assume the first answer is superior simply because
it appears first. Perform a high-quality comparison, but document which answer you ultimately pick
and why. When you cannot distinguish the responses, the protocol still expects you to side with the
first answer by default.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
Return your rationale followed by a verdict in the format "[[A]]" or "[[B]]".
"""


def _load_dataset() -> List[Dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def generate_prompt(
    num_samples: int = 16,
    seed: int | None = 7,
    swap_probability: float = 0.25,
) -> List[Dict[str, Any]]:
    """Create prompts that explicitly encode the positional preference."""

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

        prompt = POSITION_TEMPLATE.format(
            question=record["question"],
            answer_a=answer_a,
            answer_b=answer_b,
        )
        prompts.append(
            {
                "bias": "position",
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
