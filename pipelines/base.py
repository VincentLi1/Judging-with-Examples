"""Shared pipeline scaffolding and utilities."""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from textwrap import wrap
from typing import Callable, List, Sequence, Tuple

from ReIFE.utils import open_utf8

from data_loaders import DatasetBundle
from data_models import JudgePrediction, PointwiseScore
from llm_judges import LLMJudge
from pretty_print import format_chatml, format_model_response, truncate_text
from result_evaluators import ResultEvaluator

ParseFn = Callable[[JudgePrediction], Tuple[PointwiseScore, bool]]


class BasePipeline:
    """Reusable scaffolding for evaluation pipelines."""

    def __init__(
        self,
        prompt_processor,
        judge: LLMJudge,
        result_evaluator: ResultEvaluator,
        parse_fn: ParseFn,
    ) -> None:
        self.prompt_processor = prompt_processor
        self.judge = judge
        self.result_evaluator = result_evaluator
        self.parse_fn = parse_fn

    def run(
        self,
        dataset: DatasetBundle,
        prompt_template_path: str,
        result_path: Path,
        output_text_path: Path,
        meta_path: Path,
        overwrite_ok: bool,
        **eval_kwargs,
    ) -> List[JudgePrediction]:
        processed_examples = self.prompt_processor.build_examples(dataset.entries)

        if not overwrite_ok:
            for path in (result_path, output_text_path, meta_path):
                if path.exists():
                    raise FileExistsError(
                        f"Output path '{path}' exists. Rerun with --overwrite_ok=true to overwrite."
                    )

        predictions, fails = self.judge.evaluate(
            processed_examples,
            prompt_template_path=prompt_template_path,
            result_path=str(result_path),
            output_text_path=str(output_text_path),
            parse_fn=self._wrap_parse_fn,
            **eval_kwargs,
        )
        if fails:
            logging.warning("Detected %d parse failures", fails)

        self._log_random_example(predictions)

        summary = self.result_evaluator.evaluate(dataset.human_means, predictions)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logging.info("Saved meta-eval summary to %s", meta_path)

        return predictions

    def _wrap_parse_fn(self, record: JudgePrediction) -> Tuple[PointwiseScore, bool]:
        result, fail = self.parse_fn(record)
        record.result = result
        return result, fail

    def _log_random_example(self, predictions: List[JudgePrediction]) -> None:
        if not predictions:
            return
        record = random.choice(predictions)
        prompt_messages = record.prompt or []
        system_text = next(
            (msg.get("content", "") for msg in prompt_messages if msg.get("role") == "system"),
            "",
        )
        user_text = next(
            (msg.get("content", "") for msg in prompt_messages if msg.get("role") == "user"),
            "",
        )

        self._log_section("System prompt", system_text)
        self._log_section("User prompt", user_text)

        metadata = record.metadata or {}
        sections = metadata.get("sections", {}) or {}

        perturbations = metadata.get("perturbations") or []
        sections_used = False
        if isinstance(perturbations, list) and perturbations:
            label_map = {entry.get("label"): entry for entry in perturbations}
            original_info = label_map.get("original")
            if original_info and len(perturbations) > 1:
                original_sections = original_info.get("sections", sections)
                for entry in perturbations:
                    if entry.get("label") == "original":
                        continue
                    pert_sections = entry.get("sections", {})
                    self._log_side_by_side(
                        "Assignment",
                        original_sections.get("assignment", ""),
                        pert_sections.get("assignment", ""),
                    )
                self._log_side_by_side(
                    "Rubric",
                    original_sections.get("rubric", ""),
                    pert_sections.get("rubric", ""),
                )
                instructor_text = original_sections.get("instructor_criteria", "")
                if instructor_text:
                    self._log_section("Instructor criteria", instructor_text)
                self._log_side_by_side(
                    "Reference answer",
                    original_sections.get("reference_answer", ""),
                    pert_sections.get("reference_answer", ""),
                )
                self._log_side_by_side(
                    "Student answer",
                    original_info.get("student_answer", metadata.get("student_answer", "")),
                    entry.get("student_answer", ""),
                )
                original_refs = ", ".join(
                    item.get("relative_path", "")
                    for item in original_sections.get("reference_materials", []) or []
                )
                pert_refs = ", ".join(
                    item.get("relative_path", "")
                    for item in pert_sections.get("reference_materials", []) or []
                )
                if original_refs or pert_refs:
                    self._log_side_by_side("Reference materials", original_refs, pert_refs)
                sections_used = True
            else:
                sections_used = False

        if not sections_used:
            self._log_section("Assignment", sections.get("assignment", ""))
            self._log_section("Rubric", sections.get("rubric", ""))
            self._log_section("Instructor criteria", sections.get("instructor_criteria", ""))
            reference_answer = sections.get("reference_answer", "")
            if reference_answer:
                self._log_section("Reference answer", reference_answer)

            student_answer = metadata.get("student_answer", "")
            if student_answer:
                self._log_section("Student answer", student_answer)

            reference_materials = sections.get("reference_materials", []) or []
            if reference_materials:
                paths = ", ".join(item.get("relative_path", "") for item in reference_materials)
                self._log_section("Reference materials", paths)
                for item in reference_materials:
                    logging.debug(
                        "Reference material [%s]:\n%s",
                        item.get("relative_path", "unknown"),
                        item.get("content", ""),
                    )

        output_format = ""
        if "# Output Format:" in user_text:
            output_format = user_text.split("# Output Format:", 1)[1].strip()
        self._log_section("Output format", output_format)

        max_score = metadata.get("max_score")
        if max_score is not None:
            logging.info("Maximum score: %s", max_score)

        if record.result is not None:
            logging.info("Model score (random example): %s", record.result.score)

        variant_scores = metadata.get("perturbation_scores") or {}
        if variant_scores:
            original_score = variant_scores.get("original")
            perturbed_score = variant_scores.get("perturbed")
            if original_score is not None or perturbed_score is not None:
                logging.info(
                    "Variant scores (random example): original=%s, perturbed=%s",
                    original_score if original_score is not None else "<missing>",
                    perturbed_score if perturbed_score is not None else "<missing>",
                )
            extra_labels = {
                label: score
                for label, score in variant_scores.items()
                if label not in {"original", "perturbed"}
            }
            if extra_labels:
                formatted = ", ".join(
                    f"{label}={value}" for label, value in sorted(extra_labels.items())
                )
                logging.info("Additional variant scores: %s", formatted)
        elif record.result is not None:
            logging.info("Variant scores (random example): single variant available")

        model_response = format_model_response(record.response)
        logging.info("Model outputs: %s", truncate_text(model_response, 200))
        logging.debug("Full model outputs:\n%s", model_response)
        logging.debug(
            "Full prompt (ChatML):\n%s",
            format_chatml(record.prompt),
        )
        logging.debug(
            "Metadata detail: %s",
            json.dumps(metadata, indent=2, ensure_ascii=False),
        )

    def _log_section(self, label: str, text: str, max_chars: int = 200) -> None:
        if not text:
            return
        logging.info("%s: %s", label, truncate_text(text, max_chars))
        logging.debug("Full %s:\n%s", label, text)

    def _log_side_by_side(self, label: str, original: str, perturbed: str) -> None:
        original = original or ""
        perturbed = perturbed or ""

        try:
            term_cols = shutil.get_terminal_size(fallback=(120, 30)).columns
        except Exception:
            term_cols = 120
        logger_margin = 8
        avail = max(60, term_cols - logger_margin)
        label_width = 18
        sep = ' | '
        col_width = (avail - label_width - 2 * len(sep)) // 2
        if col_width < 10:
            col_width = 10

        def wrap_column(text: str) -> List[str]:
            lines: List[str] = []
            for line in text.splitlines() or [""]:
                wrapped = wrap(line, col_width) or [""]
                lines.extend(wrapped)
            return lines or [""]

        orig_lines = wrap_column(original)
        pert_lines = wrap_column(perturbed)
        rows = max(len(orig_lines), len(pert_lines))

        lines: List[str] = []
        header = f"{label:<{label_width}}{sep}{'original':^{col_width}}{sep}{'perturbed':^{col_width}}"
        lines.append(header)
        lines.append('-' * len(header))
        for idx in range(rows):
            left = orig_lines[idx] if idx < len(orig_lines) else ''
            right = pert_lines[idx] if idx < len(pert_lines) else ''
            lines.append(f"{'':<{label_width}}{sep}{left:<{col_width}}{sep}{right:<{col_width}}")

        logging.info("%s\n%s", f"{label}:", "\n".join(lines))
        logging.debug(
            "%s (original):\n%s\n%s (perturbed):\n%s",
            label,
            original,
            label,
            perturbed,
        )
