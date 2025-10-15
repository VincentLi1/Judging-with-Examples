"""Prompt processing utilities for evaluation pipelines.

This module defines a small abstraction over the string-munging logic that turns
raw dataset entries into the instructions consumed by the LLM judge.  It keeps
all prompt-shaping rules in one place so alternative processors can be plugged
in without rewriting the pipeline orchestration code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import json
import logging


def _normalize_text(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def format_score_rubric_text(rubric: Any) -> str:
    """Return a deterministic string representation of a score rubric."""

    if rubric is None:
        return ""
    if isinstance(rubric, str):
        return rubric.strip()
    if isinstance(rubric, Mapping):
        parts: List[str] = []
        description = _normalize_text(rubric.get("description"))
        if description:
            parts.append(description)

        criteria = _normalize_text(rubric.get("criteria"))
        if criteria:
            parts.append("Criteria:\n" + criteria)

        skills = rubric.get("skills")
        if isinstance(skills, Sequence):
            for item in sorted(skills, key=lambda x: str(x.get("name")) if isinstance(x, Mapping) else ""):
                if not isinstance(item, Mapping):
                    continue
                name = _normalize_text(item.get("name")) or "Skill"
                criteria_text = _normalize_text(item.get("criteria"))
                skill_section: List[str] = [f"Skill: {name}"]
                if criteria_text:
                    skill_section.append("Criteria:\n" + criteria_text)
                scale = item.get("scale")
                if isinstance(scale, Mapping) and scale:
                    ordered = sorted(scale.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))
                    scale_lines = [f"  {key}: {str(value).strip()}" for key, value in ordered]
                    skill_section.append("Scale:\n" + "\n".join(scale_lines))
                parts.append("\n".join(skill_section))

        score_keys = []
        for key in rubric.keys():
            if key.startswith("score") and key.endswith("_description"):
                try:
                    value = int(key[len("score") : key.rfind("_description")])
                except ValueError:
                    continue
                score_keys.append((value, _normalize_text(rubric[key])))
        for score, text in sorted(score_keys, key=lambda item: item[0]):
            if text:
                parts.append(f"Score {score}:\n{text}")

        scale_min = rubric.get("scale_min")
        scale_max = rubric.get("scale_max")
        if scale_min is not None or scale_max is not None:
            parts.append(
                "Scale Range: {min}–{max}".format(
                    min=scale_min if scale_min is not None else "?",
                    max=scale_max if scale_max is not None else "?",
                )
            )

        other_keys = {
            key: value
            for key, value in rubric.items()
            if key
            not in {
                "description",
                "criteria",
                "skills",
                "scale_min",
                "scale_max",
            }
            and not (key.startswith("score") and key.endswith("_description"))
        }
        if other_keys:
            parts.append("Additional Details:\n" + json.dumps(other_keys, indent=2, sort_keys=True))

        return "\n\n".join(part for part in parts if part).strip()

    return json.dumps(rubric, indent=2, sort_keys=True)


def compose_pointwise_prompt(
    instructions: str,
    reference_answer: Optional[str],
    score_rubric: Any,
    response: str,
) -> str:
    """Compose a deterministic prompt preview for pointwise judging."""

    instruction_text = _normalize_text(instructions)
    if not instruction_text:
        raise ValueError("instructions must be non-empty")
    response_text = _normalize_text(response)
    if not response_text:
        raise ValueError("response must be non-empty")

    rubric_text = format_score_rubric_text(score_rubric)

    sections: List[str] = ["### Instructions\n" + instruction_text]
    if reference_answer:
        sections.append("### Reference Answer\n" + reference_answer.strip())
    if rubric_text:
        sections.append("### Score Rubric\n" + rubric_text)
    sections.append("### Response\n" + response_text)
    return "\n\n".join(sections)


ExtraSection = Union[Tuple[str, str], Tuple[str, str, str]]

def _compose_instruction_sections(
    *,
    system_prompt: str,
    task_text: str,
    score_rubric_text: str,
    reference_answer: Optional[str] = None,
    extra_sections: Optional[Sequence[ExtraSection]] = None,
    max_score: Optional[float] = None,
    rubric_heading: str = "Score Rubric",
    instructions_heading: str = "Task Instructions",
) -> tuple[str, Dict[str, Any]]:
    sections: List[str] = []
    section_map: Dict[str, Any] = {}

    task_body = task_text.strip()
    if task_body:
        sections.append(f"### {instructions_heading}\n" + task_body)
        section_map["assignment"] = task_body

    if reference_answer:
        ref_text = reference_answer.strip()
        if ref_text:
            sections.append("### Reference Answer\n" + ref_text)
            section_map["reference_answer"] = ref_text

    rubric_body = score_rubric_text.strip()
    if rubric_body:
        sections.append(f"### {rubric_heading}\n" + rubric_body)
        section_map["rubric"] = rubric_body

    if max_score is not None:
        sections.append(f"### Maximum Score\n{max_score}")
        section_map["max_score"] = max_score

    if extra_sections:
        for section in extra_sections:
            if len(section) == 2:
                title, content = section
                map_key = title.lower().replace(" ", "_")
            else:
                title, content, map_key = section

            body = content.strip()
            if not body:
                continue
            sections.append(f"### {title}\n" + body)
            section_map[map_key] = body

    instruction_payload = f"{system_prompt}\n\n" + "\n\n".join(sections).strip()
    return instruction_payload, section_map


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
        question_field = entry.get("question", [])
        if isinstance(question_field, list):
            assignment_text = "\n".join(str(item) for item in question_field)
        else:
            assignment_text = str(question_field)

        extra_sections: List[ExtraSection] = []

        rubric_text = _normalize_text(entry.get("sample_criteria"))
        if rubric_text:
            extra_sections.append(("Rubric (Student-Facing)", rubric_text, "rubric"))

        instructor_text = _normalize_text(entry.get("tutorial_criteria"))
        if instructor_text:
            extra_sections.append(("Instructor Criteria", instructor_text, "instructor_criteria"))

        reference_materials: List[Dict[str, str]] = []
        context_blocks: List[str] = []
        for context in entry.get("context_files", []):
            rel_path = context.get("relative_path", "unknown")
            content = context.get("content", "")
            trimmed = content.strip() if isinstance(content, str) else str(content)
            context_blocks.append(f"# {rel_path}\n{trimmed}\n")
            reference_materials.append({"relative_path": rel_path, "content": trimmed})
        reference_materials_text = "\n".join(context_blocks).strip()
        if reference_materials_text:
            extra_sections.append(("Reference Materials", reference_materials_text, "reference_materials_text"))

        reference_answer: Optional[str] = None
        if self.include_reference_answer:
            sample_answer = entry.get("sample_answer")
            if isinstance(sample_answer, str) and sample_answer.strip():
                reference_answer = sample_answer

        instruction_text, section_map = _compose_instruction_sections(
            system_prompt=self.SYSTEM_INSTRUCTIONS,
            task_text=assignment_text,
            score_rubric_text="",
            reference_answer=reference_answer,
            extra_sections=extra_sections,
            max_score=entry.get("max_score"),
            instructions_heading="Assignment",
        )

        if reference_materials:
            section_map["reference_materials"] = reference_materials
            section_map.pop("reference_materials_text", None)

        return instruction_text, section_map



class BiGGenPromptProcessor(PromptProcessor):
    """Prompt processor that prepares BiGGen-Bench evaluation instructions."""

    SYSTEM_INSTRUCTIONS = (
        "You are an expert evaluator for BiGGen-Bench. Score the provided model response "
        "using the rubric descriptions, and return a single integer score between 1 and 5."
    )

    def __init__(self, include_reference_answer: bool = True) -> None:
        self.include_reference_answer = include_reference_answer

    @staticmethod
    def _format_rubric(rubric: Dict[str, object]) -> str:
        parts: List[str] = []
        criteria = rubric.get("criteria")
        if isinstance(criteria, str) and criteria.strip():
            parts.append("Criteria:\n" + criteria.strip())

        score_sections = []
        for key, value in rubric.items():
            if not key.startswith("score") or not key.endswith("_description"):
                continue
            prefix = key[len("score") : key.rfind("_description")]
            if not prefix.isdigit():
                continue
            score_val = int(prefix)
            if isinstance(value, str) and value.strip():
                score_sections.append((score_val, value.strip()))
        for score_val, description in sorted(score_sections, key=lambda item: item[0]):
            parts.append(f"Score {score_val}:\n{description}")

        return "\n\n".join(parts).strip()

    def build_examples(self, raw_entries: Iterable[dict]) -> List[PromptExample]:
        processed: List[PromptExample] = []
        for entry in raw_entries:
            instruction, section_map = self._compose_instruction(entry)
            response = entry.get("response", "")
            metadata = {
                "dataset": "biggen_bench",
                "capability": entry.get("capability"),
                "task": entry.get("task"),
                "instance_idx": entry.get("instance_idx"),
                "max_score": entry.get("max_score", 5),
                "human_scores": entry.get("human_scores", []),
                "human_score_source": entry.get("human_score_source"),
                "score_rubric": entry.get("score_rubric", {}),
                "system_prompt": entry.get("system_prompt"),
                "sections": section_map,
                "student_answer": response,
            }
            processed.append(
                PromptExample(
                    id=str(entry.get("id")),
                    instruction=instruction,
                    output=response,
                    metadata=metadata,
                )
            )
        return processed

    def _compose_instruction(self, entry: dict) -> tuple[str, Dict[str, object]]:
        instructions = _normalize_text(entry.get("input"))
        system_prompt_text = _normalize_text(entry.get("system_prompt"))

        reference_answer: Optional[str] = None
        if self.include_reference_answer:
            reference = entry.get("reference_answer")
            if isinstance(reference, str) and reference.strip():
                reference_answer = reference

        rubric = entry.get("score_rubric") or {}
        rubric_text = self._format_rubric(rubric) if isinstance(rubric, dict) else ""

        extra_sections: List[ExtraSection] = []
        if system_prompt_text:
            extra_sections.append(("System Prompt", system_prompt_text, "system_prompt"))

        instruction_text, section_map = _compose_instruction_sections(
            system_prompt=self.SYSTEM_INSTRUCTIONS,
            task_text=instructions,
            score_rubric_text=rubric_text,
            reference_answer=reference_answer,
            extra_sections=extra_sections,
            max_score=entry.get("max_score"),
        )

        return instruction_text, section_map



class FLASKPromptProcessor(PromptProcessor):
    """Prompt processor for the FLASK evaluation dataset."""

    SYSTEM_INSTRUCTIONS = (
        "You are an expert writing evaluator. Score the given response strictly according "
        "to the provided FLASK rubric and output only the numeric score."
    )

    def __init__(self, include_reference_answer: bool = True) -> None:
        self.include_reference_answer = include_reference_answer

    def build_examples(self, raw_entries: Iterable[dict]) -> List[PromptExample]:
        processed: List[PromptExample] = []
        for entry in raw_entries:
            instructions = str(entry.get("instructions") or "").strip()
            if not instructions:
                logging.warning("Skipping FLASK entry %s due to missing instructions", entry.get("id"))
                continue

            response = str(entry.get("response") or "").strip()
            if not response:
                response = "<no response available>"

            reference_answer = entry.get("reference_answer") if self.include_reference_answer else None
            rubric_text = format_score_rubric_text(entry.get("score_rubric"))

            instruction_text, section_map = _compose_instruction_sections(
                system_prompt=self.SYSTEM_INSTRUCTIONS,
                task_text=instructions,
                score_rubric_text=rubric_text,
                reference_answer=reference_answer,
                extra_sections=None,
                max_score=entry.get("max_score"),
                instructions_heading="Instructions",
            )

            metadata = {
                "dataset": "flask",
                "score_rubric": entry.get("score_rubric"),
                "score_rubric_text": rubric_text,
                "reference_answer": reference_answer,
                "max_score": entry.get("max_score", 5),
                "human_scores": entry.get("human_scores", []),
                "sections": section_map,
                "student_answer": response,
            }

            processed.append(
                PromptExample(
                    id=str(entry.get("id")),
                    instruction=instruction_text,
                    output=response,
                    metadata=metadata,
                )
            )
        return processed


class PairwiseComparisonPromptProcessor(PromptProcessor):
    """Prompt processor that compares two responses and requests a discrete ranking."""

    def __init__(self, *, system_instructions: str, scoring_scale: int) -> None:
        self.system_instructions = system_instructions
        self.scoring_scale = scoring_scale

    def build_examples(self, raw_entries: Iterable[dict]) -> List[PromptExample]:
        processed: List[PromptExample] = []
        for entry in raw_entries:
            question = str(entry.get("question") or "").strip()
            if not question:
                logging.debug("Skipping pairwise entry %s due to empty question", entry.get("id"))
                continue

            response_a = str(entry.get("response_a") or "").strip()
            response_b = str(entry.get("response_b") or "").strip()
            if not response_a or not response_b:
                logging.debug("Skipping pairwise entry %s due to missing responses", entry.get("id"))
                continue

            rubric_text = format_score_rubric_text(entry.get("score_rubric"))

            parts: List[str] = [self.system_instructions]
            parts.append(f"### Question\n{question}")
            parts.append(f"### Response A\n{response_a}")
            parts.append(f"### Response B\n{response_b}")
            if rubric_text:
                parts.append(f"### Scoring Guidance\n{rubric_text}")

            conversation_a = entry.get("conversation_a") or []
            if isinstance(conversation_a, Sequence) and conversation_a:
                formatted = []
                for turn in conversation_a:
                    if isinstance(turn, Mapping):
                        role = str(turn.get("role", "")).strip().title()
                        content = str(turn.get("content") or turn.get("text") or "").strip()
                        formatted.append(f"{role}: {content}")
                if formatted:
                    parts.append("### Conversation A\n" + "\n".join(formatted))

            conversation_b = entry.get("conversation_b") or []
            if isinstance(conversation_b, Sequence) and conversation_b:
                formatted = []
                for turn in conversation_b:
                    if isinstance(turn, Mapping):
                        role = str(turn.get("role", "")).strip().title()
                        content = str(turn.get("content") or turn.get("text") or "").strip()
                        formatted.append(f"{role}: {content}")
                if formatted:
                    parts.append("### Conversation B\n" + "\n".join(formatted))

            parts.append(
                "### Output Instructions\n"
                "Output `1` if Response A is better, `2` if Response B is better, or `0` if they are equally good. "
                "Provide only the single digit."
            )

            instruction_text = "\n\n".join(parts)

            metadata = {
                "dataset": entry.get("source_dataset"),
                "question_id": entry.get("question_id"),
                "question": question,
                "response_a": response_a,
                "response_b": response_b,
                "human_winner": entry.get("human_winner"),
                "score_rubric": entry.get("score_rubric"),
                "max_score": self.scoring_scale,
            }

            processed.append(
                PromptExample(
                    id=str(entry.get("id")),
                    instruction=instruction_text,
                    output="Not Applicable",
                    metadata=metadata,
                )
            )
        return processed


class PairwisePointwisePromptProcessor(PromptProcessor):
    """Convert pairwise preference data into pointwise scoring prompts."""

    def __init__(
        self,
        *,
        dataset_key: str,
        system_instructions: str,
        scoring_scale: int,
        include_reference_answer: bool = False,
        extra_section_builder: Optional[Callable[[dict, str], List[Tuple[str, str]]]] = None,
    ) -> None:
        self.dataset_key = dataset_key
        self.system_instructions = system_instructions
        self.SYSTEM_INSTRUCTIONS = system_instructions
        self.scoring_scale = scoring_scale
        self.include_reference_answer = include_reference_answer
        self.extra_section_builder = extra_section_builder

    def _build_extra_sections(self, entry: dict, side: str) -> List[Tuple[str, str]]:
        if self.extra_section_builder is None:
            return []
        try:
            return list(self.extra_section_builder(entry, side))
        except Exception as exc:  # pragma: no cover - defensive logging only
            logging.exception("Failed to build extra sections for %s side %s: %s", self.dataset_key, side, exc)
            return []

    def build_examples(self, raw_entries: Iterable[dict]) -> List[PromptExample]:
        processed: List[PromptExample] = []
        for entry in raw_entries:
            question_text = str(entry.get("question") or "").strip()
            if not question_text:
                logging.debug("Skipping %s entry %s due to empty question", self.dataset_key, entry.get("id"))
                continue

            rubric_text = format_score_rubric_text(entry.get("score_rubric"))
            max_score = entry.get("max_score", self.scoring_scale)

            for side in ("a", "b"):
                response_key = f"response_{side}"
                response = str(entry.get(response_key) or "").strip()
                if not response:
                    logging.debug(
                        "Skipping %s entry %s side %s due to missing response",
                        self.dataset_key,
                        entry.get("id"),
                        side,
                    )
                    continue

                extra_sections = self._build_extra_sections(entry, side)
                instruction_text, section_map = _compose_instruction_sections(
                    system_prompt=self.system_instructions,
                    task_text=question_text,
                    score_rubric_text=rubric_text,
                    reference_answer=None,
                    extra_sections=extra_sections,
                    max_score=max_score,
                    instructions_heading="Question",
                )

                metadata = {
                    "dataset": self.dataset_key,
                    "pair_id": entry.get("id"),
                    "question_id": entry.get("question_id"),
                    "side": side,
                    "other_side": "b" if side == "a" else "a",
                    "model_name": entry.get(f"model_{side}"),
                    "human_winner": entry.get("human_winner"),
                    "score_rubric_text": rubric_text,
                    "score_rubric": entry.get("score_rubric"),
                    "max_score": max_score,
                    "category": entry.get("category"),
                    "language": entry.get("language"),
                    "sections": section_map,
                    "student_answer": response,
                }

                processed.append(
                    PromptExample(
                        id=f"{entry.get('id')}-{side}",
                        instruction=instruction_text,
                        output=response,
                        metadata=metadata,
                    )
                )
        return processed


def arena_extra_sections(entry: dict, side: str) -> List[Tuple[str, str]]:
    transcript = entry.get(f"conversation_{side}") or []
    if not isinstance(transcript, Sequence):
        transcript = []
    formatted_turns: List[str] = []
    for turn in transcript:
        if not isinstance(turn, Mapping):
            continue
        role = str(turn.get("role", "")).lower()
        if not role:
            continue
        text = str(turn.get("content") or turn.get("text") or "").strip()
        formatted_turns.append(f"{role.title()}: {text}")
    if formatted_turns:
        return [("Conversation", "\n".join(formatted_turns))]
    return []


def mt_bench_extra_sections(entry: dict, side: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    category = entry.get("category")
    if isinstance(category, str) and category.strip():
        sections.append(("Category", category.strip()))
    return sections
