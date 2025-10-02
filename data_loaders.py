"""Dataset loading utilities for evaluation pipelines.

This module implements a simple strategy pattern so that additional datasets can
be plugged into the evaluation pipeline without modifying the orchestration
code.  Each loader is responsible for converting raw data into a
:class:`DatasetBundle`, which contains the entries handed off to the prompt
processor as well as the aggregate human statistics used for meta-evaluation."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Type

TEXT_EXTENSIONS = {
    ".c",
    ".h",
    ".py",
    ".s",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".cpp",
    ".hpp",
    ".java",
    ".rs",
    ".go",
    ".js",
    ".ts",
}


@dataclass
class DatasetBundle:
    """Container returned by dataset loaders."""

    entries: List[dict]
    human_means: List[float]


class DatasetLoader(ABC):
    """Strategy interface for dataset ingestion."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @abstractmethod
    def load(self) -> DatasetBundle:
        """Return the dataset bundle for the configured root."""


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name.lower() == "makefile"


def _read_file_safely(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logging.debug("Skipping non-text file: %s", path)
        return None


class LLMGraderDatasetLoader(DatasetLoader):
    """Load the entire llm_grader dataset, including tutorial code context."""

    def load(self) -> DatasetBundle:
        dataset_root = self.root
        grading_dirs = sorted(
            p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("q")
        )

        entries: List[dict] = []
        human_means: List[float] = []

        for question_dir in grading_dirs:
            question_id = question_dir.name
            grading_path = question_dir / "grading.json"
            tutorial_code_dir = dataset_root / "tutorialCode" / question_id
            tutorial_criteria_path = dataset_root / "tutorialCriteria" / f"{question_id}.json"

            with grading_path.open(encoding="utf-8") as f:
                submissions = json.load(f)
            with tutorial_criteria_path.open(encoding="utf-8") as f:
                criteria_data = json.load(f)
            criteria_payload = next(iter(criteria_data.values()))
            instructor_criteria = criteria_payload.get("criteria", "")

            context_files: List[Dict[str, str]] = []
            for path in sorted(tutorial_code_dir.rglob("*")):
                if path.is_file() and _is_text_file(path):
                    content = _read_file_safely(path)
                    if content is not None:
                        context_files.append(
                            {
                                "relative_path": str(path.relative_to(tutorial_code_dir)),
                                "content": content,
                            }
                        )

            for submission_id, payload in submissions.items():
                sample = next(iter(payload.values()))
                human_scores = [
                    sample.get(f"score_{i}")
                    for i in range(1, 4)
                    if sample.get(f"score_{i}") is not None
                ]
                if human_scores:
                    human_mean = sum(human_scores) / len(human_scores)
                else:
                    human_mean = 0.0
                human_means.append(human_mean)

                entry = {
                    "id": f"{question_id}-{submission_id}",
                    "question_id": question_id,
                    "question": sample.get("question", []),
                    "answer": sample.get("answer", ""),
                    "sample_answer": sample.get("sample_answer", ""),
                    "sample_criteria": sample.get("sample_criteria", ""),
                    "tutorial_criteria": instructor_criteria,
                    "max_score": sample.get("full_points", criteria_payload.get("full_points")),
                    "human_scores": human_scores,
                    "context_files": [dict(context) for context in context_files],
                }
                entries.append(entry)

        logging.info(
            "Loaded %d llm_grader submissions across %d questions",
            len(entries),
            len(grading_dirs),
        )
        return DatasetBundle(entries=entries, human_means=human_means)


DATASET_LOADERS: Dict[str, Type[DatasetLoader]] = {
    "llm_grader": LLMGraderDatasetLoader,
}


def register_dataset_loader(name: str, loader_cls: Type[DatasetLoader]) -> None:
    DATASET_LOADERS[name] = loader_cls


def create_dataset_loader(name: str, root: Path) -> DatasetLoader:
    try:
        loader_cls = DATASET_LOADERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset loader '{name}'") from exc
    return loader_cls(root)


def load_llm_grader_dataset(root: Path) -> DatasetBundle:
    """Convenience wrapper for existing callers."""

    return LLMGraderDatasetLoader(root).load()
