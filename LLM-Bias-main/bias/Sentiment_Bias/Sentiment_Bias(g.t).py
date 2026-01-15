"""
Created by Codex to add a ``generate_prompt`` helper for sentiment-bias probing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "extensiveDataset.json"

SENTIMENT_INSTRUCTIONS = {
    "cheerful": "bright, upbeat, and optimistic",
    "sad": "somber, empathetic, and subdued",
    "angry": "frustrated, blunt, and indignant",
    "fearful": "cautious, worried, and risk-aware",
}

SENTIMENT_TEMPLATE = """[System]
You are investigating whether judges implicitly prefer responses that match a requested tone.
For this batch, prioritize answers that feel {target_phrase}. If neither answer matches the tone,
default to the first response. Summarize your reasoning and finish with "[[A]]" or "[[B]]".
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


def _wrap_with_sentiment(text: str, tone: str) -> str:
    return f"[{tone.upper()} tone]\n{text}"


def generate_prompt(
    num_samples: int = 16,
    seed: int | None = 2024,
) -> List[Dict[str, Any]]:
    """Generate sentiment-focused prompts."""

    data = _load_dataset()
    total = len(data)
    if total == 0:
        return []

    rng = random.Random(seed)
    count = min(max(num_samples, 1), total)
    indices = rng.sample(range(total), count)
    sentiments = list(SENTIMENT_INSTRUCTIONS.keys())
    prompts: List[Dict[str, Any]] = []

    for idx in indices:
        record = data[idx]
        desired = rng.choice(sentiments)
        undesired = rng.choice([tone for tone in sentiments if tone != desired])

        answer_a = _wrap_with_sentiment(record["chosen"], desired)
        answer_b = _wrap_with_sentiment(record["rejected"], undesired)

        # Occasionally swap to keep positions mixed.
        if rng.random() < 0.4:
            answer_a, answer_b = answer_b, answer_a

        prompt = SENTIMENT_TEMPLATE.format(
            target_phrase=SENTIMENT_INSTRUCTIONS[desired],
            question=record["question"],
            answer_a=answer_a,
            answer_b=answer_b,
        )
        prompts.append(
            {
                "bias": "sentiment",
                "id": idx,
                "question": record["question"],
                "prompt": prompt,
                "target_tone": desired,
                "answer_a": answer_a,
                "answer_b": answer_b,
            }
        )

    return prompts


if __name__ == "__main__":
    sample = generate_prompt(num_samples=1)
    if sample:
        print(sample[0]["prompt"])
