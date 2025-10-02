"""Pipeline abstractions shared between different evaluation tasks."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from statistics import StatisticsError, mean, median, stdev
from textwrap import wrap
from typing import Callable, Dict, List, Tuple

import copy
import shutil

from ReIFE.utils import open_utf8

from data_loaders import DatasetBundle, create_dataset_loader
from data_models import JudgePrediction, PointwiseScore, PerturbationStatistics
from llm_judges import LLMJudge, ReIFEPointwiseJudge
from pretty_print import format_chatml, format_model_response, truncate_text
from parsers import simple_pointwise_parse
from prompt_processors import LLMGraderPromptProcessor, PromptExample
from result_evaluators import PointwiseResultEvaluator, ResultEvaluator

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
    ) -> None:
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
                # when original already logged, skip default logging
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

            reference_materials = sections.get("reference_materials", [])
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

        # Size columns to fit terminal width to avoid visual wrapping
        try:
            term_cols = shutil.get_terminal_size(fallback=(120, 30)).columns
        except Exception:
            term_cols = 120
        logger_margin = 8  # room for logging prefix like "[INFO] "
        avail = max(60, term_cols - logger_margin)
        label_width = 18
        sep = ' | '
        # total = label_width + len(sep) + col + len(sep) + col
        col_width = (avail - label_width - 2 * len(sep)) // 2
        if col_width < 10:
            col_width = 10

        from textwrap import wrap as _wrap
        def wrap_column(text: str) -> List[str]:
            lines: List[str] = []
            for line in text.splitlines() or [""]:
                wrapped = _wrap(line, col_width) or [""]
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




class SimplePromptPerturbationMixin:
    """Mixin implementing a simple two-variant prompt perturbation strategy."""

    replacement_tokens = {
        "instructions": "apple",
        "reference_answer": "banana",
        "student_answer": "capricorn",
        "rubric": "delorean",
    }

    variant_order = ("original", "perturbed")

    def _perturb_text(self, text: str, token: str) -> str:
        words = text.split()
        for idx in range(4, len(words), 5):
            words[idx] = token
        return " ".join(words)

    def _render_instruction(self, sections: Dict[str, object], max_score: float | None) -> str:
        processor = self.base.prompt_processor
        parts: List[str] = []
        assignment = sections.get("assignment")
        if assignment:
            parts.append("### Assignment\n" + assignment)
        rubric = sections.get("rubric")
        if rubric:
            parts.append("### Rubric (Student-Facing)\n" + rubric)
        instructor = sections.get("instructor_criteria")
        if instructor:
            parts.append("### Instructor Criteria\n" + instructor)
        reference_answer = sections.get("reference_answer")
        if reference_answer:
            parts.append("### Reference Answer\n" + reference_answer)
        if max_score is not None:
            parts.append(f"### Maximum Score\n{max_score}")
        reference_materials = sections.get("reference_materials", []) or []
        if reference_materials:
            blocks = [
                f"# {item.get('relative_path', 'unknown')}\n{item.get('content', '').strip()}"
                for item in reference_materials
            ]
            parts.append("### Reference Materials\n" + "\n".join(blocks).strip())
        header = f"{processor.SYSTEM_INSTRUCTIONS}\n\n" + "\n\n".join(parts).strip()
        return header

    def _build_variants(
        self, processed_examples: List[PromptExample]
    ) -> tuple[Dict[str, List[PromptExample]], Dict[str, List[Dict[str, object]]]]:
        variants: Dict[str, List[PromptExample]] = {label: [] for label in self.variant_order}
        perturbation_info: Dict[str, List[Dict[str, object]]] = {}

        for example in processed_examples:
            original_meta = copy.deepcopy(example.metadata)
            original_meta["perturbation_label"] = "original"
            variants["original"].append(
                PromptExample(
                    id=example.id,
                    instruction=example.instruction,
                    output=example.output,
                    metadata=original_meta,
                )
            )

            sections = copy.deepcopy(original_meta.get("sections", {}))
            perturbation_info[example.id] = [
                {
                    "label": "original",
                    "sections": sections,
                    "student_answer": original_meta.get("student_answer", example.output),
                }
            ]

            pert_sections = copy.deepcopy(sections)
            if "assignment" in pert_sections:
                pert_sections["assignment"] = self._perturb_text(
                    pert_sections["assignment"], self.replacement_tokens["instructions"]
                )
            if "rubric" in pert_sections:
                pert_sections["rubric"] = self._perturb_text(
                    pert_sections["rubric"], self.replacement_tokens["rubric"]
                )
            if "reference_answer" in pert_sections:
                pert_sections["reference_answer"] = self._perturb_text(
                    pert_sections["reference_answer"], self.replacement_tokens["reference_answer"]
                )

            pert_student_answer = self._perturb_text(
                original_meta.get("student_answer", example.output),
                self.replacement_tokens["student_answer"],
            )

            pert_instruction = self._render_instruction(
                pert_sections, original_meta.get("max_score")
            )

            pert_meta = copy.deepcopy(original_meta)
            pert_meta["sections"] = pert_sections
            pert_meta["student_answer"] = pert_student_answer
            pert_meta["perturbation_label"] = "perturbed"

            variants["perturbed"].append(
                PromptExample(
                    id=example.id,
                    instruction=pert_instruction,
                    output=pert_student_answer,
                    metadata=pert_meta,
                )
            )

            perturbation_info[example.id].append(
                {
                    "label": "perturbed",
                    "sections": pert_sections,
                    "student_answer": pert_student_answer,
                }
            )

        return variants, perturbation_info

    def _variant_path(self, base: Path, label: str) -> Path:
        return base.with_name(f"{base.stem}.{label}{base.suffix}")

    def _run_with_perturbation(
        self,
        dataset: DatasetBundle,
        processed_examples: List[PromptExample],
        base_result_path: Path,
        base_output_path: Path,
        base_meta_path: Path,
        prompt_template_path: str,
        overwrite_ok: bool,
        eval_options: Dict,
    ) -> None:
        variants, perturbation_info = self._build_variants(processed_examples)

        outputs_dir = base_output_path.parent
        outputs_dir.mkdir(parents=True, exist_ok=True)

        paths_to_check = [base_result_path, base_output_path, base_meta_path]
        for label in self.variant_order:
            variant_result = self._variant_path(base_result_path, label)
            variant_output = self._variant_path(base_output_path, label)
            paths_to_check.extend([variant_result, variant_output])

        if not overwrite_ok:
            for path in paths_to_check:
                if path.exists():
                    raise FileExistsError(
                        f"Output path '{path}' exists. Run with --overwrite_ok=true to overwrite."
                    )

        variant_predictions: Dict[str, List[JudgePrediction]] = {}
        variant_output_paths: Dict[str, Path] = {}

        for label in self.variant_order:
            variant_examples = variants[label]
            variant_result_path = self._variant_path(base_result_path, label)
            variant_output_path = self._variant_path(base_output_path, label)
            variant_output_paths[label] = variant_output_path

            predictions, fails = self.base.judge.evaluate(
                variant_examples,
                prompt_template_path=prompt_template_path,
                result_path=str(variant_result_path),
                output_text_path=str(variant_output_path),
                parse_fn=self.base._wrap_parse_fn,
                **eval_options,
            )
            logging.info(
                "Finished variant '%s' with %d parse failures",
                label,
                fails,
            )
            variant_predictions[label] = predictions

        id_to_variants: Dict[str, Dict[str, JudgePrediction]] = defaultdict(dict)
        for label, predictions in variant_predictions.items():
            for record in predictions:
                record_id = record.id
                id_to_variants[record_id][label] = record

        aggregated_predictions: List[JudgePrediction] = []
        all_scores: List[float] = []

        for example in processed_examples:
            record_id = example.id
            variant_map = id_to_variants.get(record_id, {})
            if not variant_map:
                continue

            per_label_scores: Dict[str, float] = {}
            scores: List[float] = []
            perturbation_records = copy.deepcopy(perturbation_info.get(record_id, []))
            label_to_record = {entry.get("label"): entry for entry in perturbation_records}

            for label in self.variant_order:
                record = variant_map.get(label)
                if record is None:
                    continue
                score = float(record.result.score if record.result is not None else 0.0)
                per_label_scores[label] = score
                scores.append(score)
                if label in label_to_record:
                    label_to_record[label]["score"] = score

            if not scores:
                continue

            median_score = median(scores)
            mean_score = mean(scores)
            try:
                std_score = stdev(scores)
            except StatisticsError:
                std_score = 0.0

            all_scores.extend(scores)

            original_record = copy.deepcopy(variant_map.get("original"))
            if original_record is None:
                original_record = copy.deepcopy(next(iter(variant_map.values())))

            metadata = copy.deepcopy(original_record.metadata or {})
            metadata["perturbation_scores"] = per_label_scores
            metadata["perturbation_stats"] = PerturbationStatistics(
                median=median_score,
                mean=mean_score,
                std=std_score,
            ).to_dict()
            metadata["perturbations"] = perturbation_records

            original_record.metadata = metadata
            original_record.result = PointwiseScore(score=median_score)
            aggregated_predictions.append(original_record)

        with open_utf8(base_result_path, "w") as f:
            for record in aggregated_predictions:
                print(json.dumps(record.to_dict()), file=f)

        if "original" in variant_output_paths:
            shutil.copyfile(variant_output_paths["original"], base_output_path)

        if all_scores:
            overall_mean = mean(all_scores)
            try:
                overall_std = stdev(all_scores)
            except StatisticsError:
                overall_std = 0.0
            logging.info(
                "Perturbation scores across variants: mean=%.2f std=%.2f", overall_mean, overall_std
            )
            logging.debug("All perturbation scores: %s", all_scores)

        summary = self.base.result_evaluator.evaluate(dataset.human_means, aggregated_predictions)
        base_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with base_meta_path.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logging.info("Saved meta-eval summary to %s", base_meta_path)

        self.base._log_random_example(aggregated_predictions)


class PointwisePipeline(SimplePromptPerturbationMixin):
    """Concrete pipeline for pointwise grading on the llm_grader dataset."""

    def __init__(
        self,
        model,
        eval_callable,
        reife_root: Path,
        model_name: str,
        dataset_loader=None,
        enable_prompt_perturbation: bool = True,
    ) -> None:
        self.reife_root = reife_root
        self.model_name = model_name
        self.prompt_method = "pointwise_vanilla"
        self.eval_method = "base_pointwise"
        prompt_processor = LLMGraderPromptProcessor()
        judge = ReIFEPointwiseJudge(model=model, eval_callable=eval_callable)
        result_evaluator = PointwiseResultEvaluator()
        self.base = BasePipeline(
            prompt_processor=prompt_processor,
            judge=judge,
            result_evaluator=result_evaluator,
            parse_fn=lambda record: simple_pointwise_parse(record, verbose=False),
        )
        dataset_root = self.reife_root.parent / "llm_grader" / "dataset_os"
        self.dataset_loader = dataset_loader or create_dataset_loader("llm_grader", dataset_root)
        self.enable_prompt_perturbation = enable_prompt_perturbation

    def run(self, overwrite_ok: bool, **eval_kwargs) -> None:
        dataset = self.dataset_loader.load()
        processed_examples = self.base.prompt_processor.build_examples(dataset.entries)

        results_dir = self.reife_root / "results"
        outputs_dir = results_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        result_path = results_dir / self._result_filename()
        output_text_path = outputs_dir / self._output_text_filename()
        meta_path = results_dir / self._meta_filename()

        eval_options = {
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "max_tokens": 8,
            "logprobs": None,
            "batch_size": 1,
            "instruction_marker": "INSTRUCTION",
            "output_marker": "OUTPUT",
        }
        eval_options.update(eval_kwargs)

        prompt_template_path = str(
            self.reife_root / "prompts" / f"{self.prompt_method}.txt"
        )

        if self.enable_prompt_perturbation:
            self._run_with_perturbation(
                dataset=dataset,
                processed_examples=processed_examples,
                base_result_path=result_path,
                base_output_path=output_text_path,
                base_meta_path=meta_path,
                prompt_template_path=prompt_template_path,
                overwrite_ok=overwrite_ok,
                eval_options=eval_options,
            )
        else:
            self.base.run(
                dataset=dataset,
                prompt_template_path=prompt_template_path,
                result_path=result_path,
                output_text_path=output_text_path,
                meta_path=meta_path,
                overwrite_ok=overwrite_ok,
                **eval_options,
            )

    def _result_filename(self) -> str:
        dataset_name = "llm_grader_pointwise"
        return f"{dataset_name}.{self.model_name}.{self.eval_method}.{self.prompt_method}.jsonl"

    def _output_text_filename(self) -> str:
        dataset_name = "llm_grader_pointwise"
        return f"{dataset_name}.{self.model_name}.{self.eval_method}.{self.prompt_method}.txt"

    def _meta_filename(self) -> str:
        dataset_name = "llm_grader_pointwise"
        return f"{dataset_name}.{self.model_name}.{self.eval_method}.{self.prompt_method}.meta.json"
