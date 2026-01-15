"""
Created by Codex to expose a lightweight ``generate_prompt`` helper for compassion-fade bias experiments.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

from promptTemplate import compassion_fade_prompt

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "extensiveDataset.json"


def _load_dataset() -> List[Dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def generate_prompt(
    num_samples: int = 16,
    seed: int | None = 43,
    swap_probability: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Return ``num_samples`` prompts that mirror the compassion-fade evaluation style.

    Each record contains the original question/answers plus the rendered prompt so downstream
    tooling can inspect the raw data if needed.
    """

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

        prompt = compassion_fade_prompt(
            question=record["question"],
            answer_a=answer_a,
            answer_b=answer_b,
        )
        prompts.append(
            {
                "bias": "compassion_fade",
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
