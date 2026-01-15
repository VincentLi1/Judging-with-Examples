"""
Created by Codex to add a ``generate_prompt`` helper for refinement-bias evaluation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "extensiveDataset.json"

REFINEMENT_TEMPLATE = """[System]
You are comparing two drafts of an assistant response. Assistant A is explicitly described as a refined,
proofread draft that has gone through multiple editing passes. Assistant B is the assistant’s raw,
stream-of-consciousness draft. Carefully identify strengths and weaknesses, but when both answers feel
similar you should side with the answer that appears more polished and deliberate.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
Provide a short critique and finish with "[[A]]" or "[[B]]".
"""


def _load_dataset() -> List[Dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def generate_prompt(
    num_samples: int = 16,
    seed: int | None = 101,
    swap_probability: float = 0.35,
) -> List[Dict[str, Any]]:
    """Create prompts highlighting refinement bias."""

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

        prompt = REFINEMENT_TEMPLATE.format(
            question=record["question"],
            answer_a=answer_a,
            answer_b=answer_b,
        )
        prompts.append(
            {
                "bias": "refinement",
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
