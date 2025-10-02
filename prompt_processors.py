"""Prompt processing utilities for evaluation pipelines.

This module defines a small abstraction over the string-munging logic that turns
raw dataset entries into the instructions consumed by the LLM judge.  It keeps
all prompt-shaping rules in one place so alternative processors can be plugged
in without rewriting the pipeline orchestration code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict


@dataclass
class PromptExample:
    """Container describing a processed prompt ready for formatting.

    Attributes
    ----------
    id:
        Unique identifier for the example (e.g., ``q3-17``).
    instruction:
        The fully composed instruction string that will replace the
        ``{INSTRUCTION}`` placeholder in the template.
    output:
        The student answer or model output that should be scored.
    metadata:
        Optional dictionary carrying additional information used for logging or
        downstream analysis.
    """

    id: str
    instruction: str
    output: str
    metadata: Dict[str, object]


class PromptProcessor(ABC):
    """Base interface for prompt processors."""

    @abstractmethod
    def build_examples(self, raw_entries: Iterable[dict]) -> List[PromptExample]:
        """Transform raw dataset entries into :class:`PromptExample` objects."""


class LLMGraderPromptProcessor(PromptProcessor):
    """Prompt processor that assembles instructions for the llm_grader dataset."""

    SYSTEM_INSTRUCTIONS = (
        "You are a meticulous teaching assistant who grades student responses "
        "according to the provided rubric and reference materials."
    )

    def __init__(self, include_reference_answer: bool = True) -> None:
        self.include_reference_answer = include_reference_answer

    def build_examples(self, raw_entries: Iterable[dict]) -> List[PromptExample]:
        processed: List[PromptExample] = []
        for entry in raw_entries:
            instruction, section_map = self._compose_instruction(entry)
            metadata = {
                "question_id": entry.get("question_id"),
                "max_score": entry.get("max_score"),
                "human_scores": entry.get("human_scores", []),
                "context_files": entry.get("context_files", []),
                "sections": section_map,
                "student_answer": entry.get("answer", ""),
            }
            processed.append(
                PromptExample(
                    id=entry["id"],
                    instruction=instruction,
                    output=entry.get("answer", ""),
                    metadata=metadata,
                )
            )
        return processed

    def _compose_instruction(self, entry: dict) -> tuple[str, Dict[str, object]]:
        sections: List[str] = []
        section_map: Dict[str, object] = {}

        question_field = entry.get("question", [])
        if isinstance(question_field, list):
            question_text = "\n".join(question_field)
        else:
            question_text = str(question_field)
        assignment_text = question_text.strip()
        sections.append("### Assignment\n" + assignment_text)
        section_map["assignment"] = assignment_text

        rubric = entry.get("sample_criteria") or ""
        if rubric:
            rubric_text = rubric.strip()
            sections.append("### Rubric (Student-Facing)\n" + rubric_text)
            section_map["rubric"] = rubric_text

        criteria = entry.get("tutorial_criteria") or ""
        if criteria:
            criteria_text = criteria.strip()
            sections.append("### Instructor Criteria\n" + criteria_text)
            section_map["instructor_criteria"] = criteria_text

        if self.include_reference_answer:
            reference = entry.get("sample_answer") or ""
            if reference:
                reference_text = reference.strip()
                sections.append("### Reference Answer\n" + reference_text)
                section_map["reference_answer"] = reference_text

        max_score = entry.get("max_score")
        if max_score is not None:
            sections.append(f"### Maximum Score\n{max_score}")
            section_map["max_score"] = max_score

        context_blocks: List[str] = []
        reference_materials: List[Dict[str, str]] = []
        for context in entry.get("context_files", []):
            rel_path = context.get("relative_path", "unknown")
            content = context.get("content", "")
            trimmed = content.strip()
            context_blocks.append(f"# {rel_path}\n{trimmed}\n")
            reference_materials.append({"relative_path": rel_path, "content": trimmed})
        if context_blocks:
            ref_text = "\n".join(context_blocks).strip()
            sections.append("### Reference Materials\n" + ref_text)
            section_map["reference_materials"] = reference_materials

        header = f"{self.SYSTEM_INSTRUCTIONS}\n\n" + "\n\n".join(sections).strip()
        return header, section_map
