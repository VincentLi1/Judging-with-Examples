"""Pointwise evaluation pipeline implementation."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from statistics import StatisticsError, mean, median, stdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ReIFE.utils import open_utf8

from data_loaders import DatasetLoader, create_dataset_loader
from data_models import JudgePrediction, PointwiseScore, PerturbationStatistics
from llm_judges import ReIFEPointwiseJudge
from parsers import simple_pointwise_parse
from prompt_processors import (
    BiGGenPromptProcessor,
    FLASKPromptProcessor,
    LLMGraderPromptProcessor,
    PairwiseComparisonPromptProcessor,
    PairwisePointwisePromptProcessor,
    PromptExample,
    PromptProcessor,
    arena_extra_sections,
    mt_bench_extra_sections,
)
from result_evaluators import PointwiseResultEvaluator

from .base import BasePipeline


@dataclass
class DatasetPipelineConfig:
    """Configuration container for running an individual dataset."""

    key: str
    loader: DatasetLoader
    prompt_processor: PromptProcessor
    prompt_template: Path
    result_basename: str
    output_basename: str
    meta_basename: str
    sample_basename: str


class SimplePromptPerturbationMixin:
    """Mixin implementing a simple two-variant prompt perturbation strategy."""

    replacement_tokens = {
        "instructions": "apple",
        "reference_answer": "banana",
        "student_answer": "capricorn",
        "rubric": "delorean",
    }

    variant_order = ("original", "perturbed")

    def _perturb_text(self, text: str | None, token: str) -> str:
        if not text:
            return text or ""
        words = text.split()
        for idx in range(4, len(words), 5):
            words[idx] = token
        return " ".join(words)

    def _render_instruction(self, sections: Dict[str, object], max_score: float | None) -> str:
        processor = self.base.prompt_processor  # type: ignore[attr-defined]
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
        return f"{processor.SYSTEM_INSTRUCTIONS}\n\n" + "\n\n".join(parts).strip()

    def _build_variants(
        self, processed_examples: List[PromptExample]
    ) -> tuple[Dict[str, List[PromptExample]], Dict[str, List[Dict[str, object]]]]:
        variants: Dict[str, List[PromptExample]] = {label: [] for label in self.variant_order}
        perturbation_info: Dict[str, List[Dict[str, object]]] = {}

        for example in processed_examples:
            original_meta = copy.deepcopy(example.metadata)
            original_meta["perturbation_label"] = "original"
            original_student_answer = original_meta.get("student_answer") or example.output
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
                    "student_answer": original_student_answer,
                }
            ]

            pert_sections = copy.deepcopy(sections)
            if "assignment" in pert_sections:
                assignment_text = pert_sections["assignment"]
                if isinstance(assignment_text, str) and assignment_text.strip():
                    pert_sections["assignment"] = self._perturb_text(
                        assignment_text, self.replacement_tokens["instructions"]
                    )
            if "rubric" in pert_sections:
                rubric_text = pert_sections["rubric"]
                if isinstance(rubric_text, str) and rubric_text.strip():
                    pert_sections["rubric"] = self._perturb_text(
                        rubric_text, self.replacement_tokens["rubric"]
                    )
            if "reference_answer" in pert_sections:
                reference_text = pert_sections["reference_answer"]
                if isinstance(reference_text, str) and reference_text.strip():
                    pert_sections["reference_answer"] = self._perturb_text(
                        reference_text, self.replacement_tokens["reference_answer"]
                    )

            pert_student_answer = self._perturb_text(
                original_student_answer,
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
        dataset,
        processed_examples: List[PromptExample],
        base_result_path: Path,
        base_output_path: Path,
        base_meta_path: Path,
        prompt_template_path: str,
        overwrite_ok: bool,
        eval_options: Dict,
        *,
        dataset_key: Optional[str] = None,
    ) -> List[JudgePrediction]:
        """Run the pointwise judge across original and perturbed prompts."""

        if dataset_key:
            logging.debug("Running perturbation workflow for dataset %s", dataset_key)

        variants, perturbation_info = self._build_variants(processed_examples)

        variant_paths = self._prepare_output_paths(
            base_result_path=base_result_path,
            base_output_path=base_output_path,
            base_meta_path=base_meta_path,
            overwrite_ok=overwrite_ok,
        )

        variant_predictions, variant_output_paths = self._evaluate_variants(
            variants=variants,
            variant_paths=variant_paths,
            prompt_template_path=prompt_template_path,
            eval_options=eval_options,
        )

        aggregated_predictions, all_scores = self._aggregate_predictions(
            processed_examples=processed_examples,
            variant_predictions=variant_predictions,
            perturbation_info=perturbation_info,
        )

        self._persist_perturbation_outputs(
            base_result_path=base_result_path,
            base_output_path=base_output_path,
            base_meta_path=base_meta_path,
            aggregated_predictions=aggregated_predictions,
            all_scores=all_scores,
            human_means=dataset.human_means,
            variant_output_paths=variant_output_paths,
        )

        self.base._log_random_example(aggregated_predictions)  # type: ignore[attr-defined]
        return aggregated_predictions

    def _prepare_output_paths(
        self,
        *,
        base_result_path: Path,
        base_output_path: Path,
        base_meta_path: Path,
        overwrite_ok: bool,
    ) -> Dict[str, Tuple[Path, Path]]:
        """Create variant-specific paths and guard against unintended overwrites."""

        outputs_dir = base_output_path.parent
        outputs_dir.mkdir(parents=True, exist_ok=True)

        variant_paths: Dict[str, Tuple[Path, Path]] = {}
        paths_to_check = [base_result_path, base_output_path, base_meta_path]

        for label in self.variant_order:
            variant_result = self._variant_path(base_result_path, label)
            variant_output = self._variant_path(base_output_path, label)
            variant_paths[label] = (variant_result, variant_output)
            paths_to_check.extend([variant_result, variant_output])

        if not overwrite_ok:
            for path in paths_to_check:
                if path.exists():
                    raise FileExistsError(
                        f"Output path '{path}' exists. Run with --overwrite_ok=true to overwrite."
                    )

        return variant_paths

    def _evaluate_variants(
        self,
        *,
        variants: Dict[str, List[PromptExample]],
        variant_paths: Dict[str, Tuple[Path, Path]],
        prompt_template_path: str,
        eval_options: Dict,
    ) -> Tuple[Dict[str, List[JudgePrediction]], Dict[str, Path]]:
        """Run the judge for each perturbation variant and collect predictions."""

        variant_predictions: Dict[str, List[JudgePrediction]] = {}
        variant_output_paths: Dict[str, Path] = {}

        for label in self.variant_order:
            variant_examples = variants.get(label, [])
            if label not in variant_paths:
                continue
            result_path, output_path = variant_paths[label]
            predictions, fails = self.base.judge.evaluate(  # type: ignore[attr-defined]
                variant_examples,
                prompt_template_path=prompt_template_path,
                result_path=str(result_path),
                output_text_path=str(output_path),
                parse_fn=self.base._wrap_parse_fn,  # type: ignore[attr-defined]
                **eval_options,
            )
            logging.info("Finished variant '%s' with %d parse failures", label, fails)
            variant_predictions[label] = predictions
            variant_output_paths[label] = output_path

        return variant_predictions, variant_output_paths

    def _aggregate_predictions(
        self,
        *,
        processed_examples: List[PromptExample],
        variant_predictions: Dict[str, List[JudgePrediction]],
        perturbation_info: Dict[str, List[Dict[str, object]]],
    ) -> Tuple[List[JudgePrediction], List[float]]:
        """Combine per-variant predictions into a single aggregated record for each example."""

        id_to_variants: Dict[str, Dict[str, JudgePrediction]] = defaultdict(dict)
        for label, predictions in variant_predictions.items():
            for record in predictions:
                id_to_variants[record.id][label] = record

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

        return aggregated_predictions, all_scores

    def _persist_perturbation_outputs(
        self,
        *,
        base_result_path: Path,
        base_output_path: Path,
        base_meta_path: Path,
        aggregated_predictions: List[JudgePrediction],
        all_scores: List[float],
        human_means: List[float],
        variant_output_paths: Mapping[str, Path],
    ) -> None:
        """Persist aggregated predictions, copy outputs, and write meta summaries."""

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

        summary = self.base.result_evaluator.evaluate(  # type: ignore[attr-defined]
            human_means, aggregated_predictions
        )
        base_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with base_meta_path.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)


class PointwisePipeline(SimplePromptPerturbationMixin):
    """Concrete pointwise pipeline capable of orchestrating multiple datasets."""

    DEFAULT_EVAL_OPTIONS: Dict[str, object] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": 8,
        "logprobs": None,
        "batch_size": 1,
        "instruction_marker": "INSTRUCTION",
        "output_marker": "OUTPUT",
    }

    def __init__(
        self,
        model,
        eval_callable,
        reife_root: Path,
        model_name: str,
        dataset_loaders: Optional[Dict[str, DatasetLoader]] = None,
        datasets: Optional[Sequence[str]] = None,
        enable_prompt_perturbation: bool = True,
        results_root: Optional[Path] = None,
        logs_root: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.eval_callable = eval_callable
        self.reife_root = reife_root
        self.model_name = model_name
        self.prompt_method = "pointwise_vanilla"
        self.eval_method = "base_pointwise"
        self.enable_prompt_perturbation = enable_prompt_perturbation

        self.results_root = (results_root or (self.reife_root / "results")).resolve()
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.logs_root = (logs_root or (self.reife_root / "logs")).resolve()
        self.logs_root.mkdir(parents=True, exist_ok=True)

        self.datasets = list(datasets) if datasets else None
        self.dataset_configs = self._initialise_dataset_configs(dataset_loaders or {})

        primary_config = self.dataset_configs.get("llm_grader")
        if primary_config is None and self.dataset_configs:
            primary_config = next(iter(self.dataset_configs.values()))

        # Base pipeline instance is re-created per dataset while exposing a sensible default.
        self.base: BasePipeline | None = None
        if primary_config is not None:
            self._configure_base_pipeline(primary_config)


    def _dataset_definitions(
        self, overrides: Dict[str, DatasetLoader]
    ) -> Dict[str, Tuple[DatasetLoader, PromptProcessor, Path]]:
        project_root = self.reife_root.parent
        prompt_template = self.reife_root / "prompts" / f"{self.prompt_method}.txt"

        definitions: Dict[str, Tuple[DatasetLoader, PromptProcessor, Path]] = {}

        llm_loader = overrides.get("llm_grader") or create_dataset_loader(
            "llm_grader", project_root / "llm_grader" / "dataset_os"
        )
        definitions["llm_grader"] = (llm_loader, LLMGraderPromptProcessor(), prompt_template)

        biggen_loader = overrides.get("biggen_bench") or create_dataset_loader(
            "biggen_bench", project_root / "prometheus-eval" / "BiGGen-Bench"
        )
        definitions["biggen_bench"] = (biggen_loader, BiGGenPromptProcessor(), prompt_template)

        flask_loader = overrides.get("flask") or create_dataset_loader(
            "flask", project_root / "FLASK"
        )
        definitions["flask"] = (flask_loader, FLASKPromptProcessor(), prompt_template)

        mt_loader = overrides.get("mt_bench") or create_dataset_loader("mt_bench", project_root)
        if hasattr(mt_loader, "scoring_scale"):
            mt_loader.scoring_scale = 5
        mt_scale = getattr(mt_loader, "scoring_scale", 5)
        mt_processor = PairwisePointwisePromptProcessor(
            dataset_key="mt_bench",
            system_instructions=(
                "You are an MT-Bench adjudicator. Score the assistant response on a {scale}-point scale (1 = lowest, {scale} = highest) "
                "considering helpfulness, accuracy, and clarity."
            ).format(scale=mt_scale),
            scoring_scale=mt_scale,
            extra_section_builder=mt_bench_extra_sections,
        )
        definitions["mt_bench"] = (mt_loader, mt_processor, prompt_template)

        arena_loader = overrides.get("chatbot_arena") or create_dataset_loader("chatbot_arena", project_root)
        arena_processor = PairwiseComparisonPromptProcessor(
            system_instructions=(
                "You are comparing two Chatbot Arena responses. After reading both, output `1` if Response A is better, "
                "`2` if Response B is better, or `0` if they are equally strong."
            ),
            scoring_scale=2,
        )
        definitions["chatbot_arena"] = (arena_loader, arena_processor, prompt_template)

        return definitions

    def _initialise_dataset_configs(
        self, overrides: Dict[str, DatasetLoader]
    ) -> Dict[str, DatasetPipelineConfig]:
        configs: Dict[str, DatasetPipelineConfig] = {}
        definitions = self._dataset_definitions(overrides)
        for key, (loader, processor, template) in definitions.items():
            configs[key] = self._make_config(key, loader, processor, template)
        for key, loader in overrides.items():
            if key in configs:
                continue
            logging.warning(
                "No built-in prompt processor for dataset '%s'; provide a PromptProcessor to run it.",
                key,
            )
        return configs

    def _make_config(
        self,
        key: str,
        loader: DatasetLoader,
        prompt_processor: PromptProcessor,
        prompt_template: Path,
    ) -> DatasetPipelineConfig:
        return DatasetPipelineConfig(
            key=key,
            loader=loader,
            prompt_processor=prompt_processor,
            prompt_template=prompt_template,
            result_basename=self._result_basename(key),
            output_basename=self._output_basename(key),
            meta_basename=self._meta_basename(key),
            sample_basename=self._sample_basename(key),
        )

    def _result_basename(self, dataset_key: str) -> str:
        return f"{dataset_key}_{self.prompt_method}.{self.eval_method}.{self.model_name}.jsonl"

    def _output_basename(self, dataset_key: str) -> str:
        return f"{dataset_key}_{self.prompt_method}.{self.eval_method}.{self.model_name}.txt"

    def _meta_basename(self, dataset_key: str) -> str:
        return f"{dataset_key}_{self.prompt_method}.{self.eval_method}.{self.model_name}.meta.json"

    def _sample_basename(self, dataset_key: str) -> str:
        return f"{dataset_key}_{self.model_name}_prompt_samples.json"

    def _dataset_dir(self, dataset_key: str) -> Path:
        dataset_dir = self.results_root / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _build_base_pipeline(self, prompt_processor: PromptProcessor) -> BasePipeline:
        judge = ReIFEPointwiseJudge(model=self.model, eval_callable=self.eval_callable)
        result_evaluator = PointwiseResultEvaluator()
        return BasePipeline(
            prompt_processor=prompt_processor,
            judge=judge,
            result_evaluator=result_evaluator,
            parse_fn=lambda record: simple_pointwise_parse(record, verbose=False),
        )

    def _configure_base_pipeline(self, config: DatasetPipelineConfig) -> None:
        """Ensure ``self.base`` is ready for the current dataset."""

        if self.base is None:
            self.base = self._build_base_pipeline(config.prompt_processor)
        else:
            self.base.prompt_processor = config.prompt_processor
            self.base.judge = ReIFEPointwiseJudge(model=self.model, eval_callable=self.eval_callable)
            self.base.result_evaluator = PointwiseResultEvaluator()
            self.base.parse_fn = lambda record: simple_pointwise_parse(record, verbose=False)

    def _write_prompt_samples(
        self,
        *,
        dataset_key: str,
        examples: Sequence[PromptExample],
        config: DatasetPipelineConfig,
        output_dir: Path,
    ) -> None:
        """Persist a small prompt sample for debugging and audits."""

        if not examples:
            return
        sample_path = output_dir / config.sample_basename
        preview = [
            {"id": example.id, "instruction": example.instruction, "output": example.output}
            for example in examples[: min(3, len(examples))]
        ]
        with sample_path.open("w", encoding="utf-8") as handle:
            json.dump(preview, handle, indent=2)

    def _postprocess_dataset(
        self,
        *,
        dataset_key: str,
        config: DatasetPipelineConfig,
        dataset,
        examples: Sequence[PromptExample],
        predictions: Sequence[JudgePrediction],
        result_path: Path,
        meta_path: Path,
    ) -> None:
        """Hook for subclasses; default implementation is a no-op."""
        _ = (dataset_key, config, dataset, examples, predictions, result_path, meta_path)

    def _result_filename(self, dataset_key: str) -> str:
        return self.dataset_configs[dataset_key].result_basename

    def _output_text_filename(self, dataset_key: str) -> str:
        return self.dataset_configs[dataset_key].output_basename

    def _meta_filename(self, dataset_key: str) -> str:
        return self.dataset_configs[dataset_key].meta_basename

    def run(self, overwrite_ok: bool, **eval_kwargs) -> None:
        datasets = self.datasets or list(self.dataset_configs.keys())
        for dataset_key in datasets:
            config = self.dataset_configs.get(dataset_key)
            if config is None:
                logging.warning("Dataset '%s' is not configured; skipping.", dataset_key)
                continue

            dataset_dir = self._dataset_dir(dataset_key)
            dataset = config.loader.load()
            examples = config.prompt_processor.build_examples(dataset.entries)

            self._configure_base_pipeline(config)

            self._write_prompt_samples(
                dataset_key=dataset_key,
                examples=examples,
                config=config,
                output_dir=dataset_dir,
            )

            eval_options = dict(self.DEFAULT_EVAL_OPTIONS)
            eval_options.update(eval_kwargs)

            result_path = dataset_dir / config.result_basename
            output_text_path = dataset_dir / config.output_basename
            meta_path = dataset_dir / config.meta_basename

            predictions: List[JudgePrediction]
            if self.enable_prompt_perturbation:
                predictions = self._run_with_perturbation(
                    dataset=dataset,
                    processed_examples=examples,
                    base_result_path=result_path,
                    base_output_path=output_text_path,
                    base_meta_path=meta_path,
                    prompt_template_path=str(config.prompt_template),
                    overwrite_ok=overwrite_ok,
                    eval_options=eval_options,
                    dataset_key=dataset_key,
                )
            else:
                predictions = self.base.run(  # type: ignore[assignment]
                    dataset=dataset,
                    prompt_template_path=str(config.prompt_template),
                    result_path=result_path,
                    output_text_path=output_text_path,
                    meta_path=meta_path,
                    overwrite_ok=overwrite_ok,
                    **eval_options,
                )

            self._postprocess_dataset(
                dataset_key=dataset_key,
                config=config,
                dataset=dataset,
                examples=examples,
                predictions=predictions,
                result_path=result_path,
                meta_path=meta_path,
            )
