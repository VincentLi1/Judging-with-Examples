"""Unit tests for the refactored pointwise pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
import types
import tempfile
from unittest import mock
import argparse

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = types.SimpleNamespace()

import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

REIFE_DIR = MODULE_DIR / "ReIFE"
if str(REIFE_DIR) not in sys.path:
    sys.path.append(str(REIFE_DIR))

from parsers import simple_pointwise_parse  # noqa: E402
from pipelines import PointwisePipeline, SimplePromptPerturbationMixin  # noqa: E402
from result_evaluators import PointwiseResultEvaluator  # noqa: E402
import pointwise_pipeline  # noqa: E402
from prompt_processors import PromptExample  # noqa: E402
from data_models import JudgePrediction, PointwiseScore  # noqa: E402
from data_loaders import BiGGenDatasetLoader, DatasetBundle  # noqa: E402
from metrics import (  # noqa: E402
    analyze_pointwise_results,
    continuous_robustness,
    var_std_under_perturbation,
)


def _artifact_paths(pipeline: PointwisePipeline, dataset_prefix: str) -> tuple[Path, Path, Path]:
    results_dir = pipeline.reife_root / "results"
    outputs_dir = results_dir / "outputs"
    result_path = results_dir / pipeline._result_filename(dataset_prefix)
    output_text_path = outputs_dir / pipeline._output_text_filename(dataset_prefix)
    meta_path = results_dir / pipeline._meta_filename(dataset_prefix)
    return result_path, output_text_path, meta_path


class DummyModelTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.model, _ = pointwise_pipeline.create_model(True)
        except Exception as exc:
            self.skipTest(f"Dummy model unavailable: {exc}")

    def test_create_model_namespace_dummy(self) -> None:
        namespace = argparse.Namespace(use_dummy_model=True)
        model, name = pointwise_pipeline.create_model(namespace)
        self.assertEqual(name, "dummy_api")
        self.assertTrue(hasattr(model, "generate"))

    def test_keyword_point_allocation(self) -> None:
        score_none = self.model.debug_score("irrelevant text")
        self.assertEqual(score_none, 0)

        rich_answer = self.model.debug_score(
            "SJF and FIFO update the response order and turnaround stats"
        )
        self.assertEqual(rich_answer, 19)


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)
        self.pipeline = PointwisePipeline(
            model=object(),
            eval_callable=None,
            reife_root=pointwise_pipeline.PROJECT_ROOT / "ReIFE",
            model_name="dummy_model",
            datasets=["llm_grader"],
        )

        self.bundle = DatasetBundle(
            entries=[
                {
                    "id": "llm-1",
                    "question_id": "q1",
                    "max_score": 5,
                    "human_scores": [],
                }
            ],
            human_means=[0.0],
        )
        self.examples = [
            PromptExample(
                id="llm-1",
                instruction="Grade this response",
                output="Sample response",
                metadata={
                    "sections": {
                        "assignment": "Grade this response",
                        "rubric": "Score 1: poor\nScore 5: great",
                        "reference_answer": "gold",
                    },
                    "student_answer": "Sample response",
                    "question_id": "q1",
                    "max_score": 5,
                },
            )
        ]
        config = self.pipeline.dataset_configs["llm_grader"]
        config.loader = mock.Mock()
        config.loader.load.return_value = self.bundle
        config.prompt_processor = mock.Mock()
        config.prompt_processor.build_examples.return_value = self.examples

        self.expected_dataset_dir = self.pipeline.results_root / "llm_grader"
        self.expected_result_path = self.expected_dataset_dir / config.result_basename

    def test_pipeline_run_invokes_perturbation_flow(self) -> None:
        captured: dict = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return []

        with mock.patch.object(self.pipeline, "_write_prompt_samples"), mock.patch.object(
            self.pipeline, "_postprocess_dataset"
        ) as postprocess_mock, mock.patch.object(
            self.pipeline, "_run_with_perturbation", side_effect=fake_run
        ) as patched:
            self.pipeline.run(overwrite_ok=True)

        patched.assert_called_once()
        self.assertIn("dataset", captured)
        self.assertEqual(captured["base_result_path"], self.expected_result_path)
        postprocess_mock.assert_called_once()

    def test_pipeline_run_without_perturbation_uses_base(self) -> None:
        pipeline = PointwisePipeline(
            model=object(),
            eval_callable=None,
            reife_root=pointwise_pipeline.PROJECT_ROOT / "ReIFE",
            model_name="dummy_model",
            enable_prompt_perturbation=False,
            datasets=["llm_grader"],
        )

        bundle = DatasetBundle(
            entries=[
                {
                    "id": "llm-1",
                    "question_id": "q1",
                    "max_score": 5,
                    "human_scores": [],
                }
            ],
            human_means=[0.0],
        )
        examples = [
            PromptExample(
                id="llm-1",
                instruction="Grade this response",
                output="Sample response",
                metadata={
                    "sections": {
                        "assignment": "Grade this response",
                        "rubric": "Score 1: poor\nScore 5: great",
                    },
                    "student_answer": "Sample response",
                    "question_id": "q1",
                    "max_score": 5,
                },
            )
        ]
        config = pipeline.dataset_configs["llm_grader"]
        config.loader = mock.Mock()
        config.loader.load.return_value = bundle
        config.prompt_processor = mock.Mock()
        config.prompt_processor.build_examples.return_value = examples

        called = {}

        def fake_base_run(dataset, **_):
            called["dataset"] = dataset
            return []

        with mock.patch.object(pipeline, "_write_prompt_samples"), mock.patch.object(
            pipeline, "_postprocess_dataset"
        ) as postprocess_mock, mock.patch.object(
            pipeline, "_run_with_perturbation"
        ) as patch_perturb, mock.patch.object(
            pipeline.base, "run", side_effect=fake_base_run
        ) as patch_base:
            pipeline.run(overwrite_ok=True)

        patch_perturb.assert_not_called()
        patch_base.assert_called_once()
        self.assertIn("dataset", called)
        postprocess_mock.assert_called_once()


class MultiDatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)
        biggen_root = pointwise_pipeline.PROJECT_ROOT / "prometheus-eval" / "BiGGen-Bench"
        truncated_biggen_loader = BiGGenDatasetLoader(biggen_root, max_examples=5)
        self.pipeline = PointwisePipeline(
            model=object(),
            eval_callable=None,
            reife_root=pointwise_pipeline.PROJECT_ROOT / "ReIFE",
            model_name="dummy_model",
            dataset_loaders={"biggen_bench": truncated_biggen_loader},
        )
        self.calls: list[str] = []

        bundle = DatasetBundle(
            entries=[{"id": "stub", "question_id": "q", "max_score": 5, "human_scores": []}],
            human_means=[0.0],
        )
        examples = [
            PromptExample(
                id="stub",
                instruction="Evaluate",
                output="Answer",
                metadata={
                    "sections": {
                        "assignment": "Evaluate",
                        "rubric": "Score 1: bad\nScore 5: good",
                    },
                    "student_answer": "Answer",
                },
            )
        ]
        for config in self.pipeline.dataset_configs.values():
            config.loader = mock.Mock()
            config.loader.load.return_value = bundle
            config.prompt_processor = mock.Mock()
            config.prompt_processor.build_examples.return_value = examples

    def test_all_datasets_are_processed(self) -> None:
        def fake_run(**kwargs):
            dataset_key = kwargs.get("dataset_key")
            if dataset_key:
                self.calls.append(dataset_key)
            else:
                result_path = kwargs.get("base_result_path")
                if result_path is not None:
                    self.calls.append(result_path.parent.name)
            return []

        def fake_postprocess(**kwargs):
            config = kwargs.get("config")
            if config is not None:
                self.calls.append(config.key)

        with mock.patch.object(self.pipeline, "_write_prompt_samples"), mock.patch.object(
            self.pipeline, "_postprocess_dataset", side_effect=fake_postprocess
        ), mock.patch.object(
            self.pipeline, "_run_with_perturbation", side_effect=fake_run
        ):
            self.pipeline.run(overwrite_ok=True)

        dataset_keys = {key for key in self.calls if key in {"llm_grader", "biggen_bench"}}
        self.assertSetEqual(dataset_keys, {"llm_grader", "biggen_bench"})


class BiGGenDatasetLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        root = pointwise_pipeline.PROJECT_ROOT / "prometheus-eval" / "BiGGen-Bench"
        self.loader = BiGGenDatasetLoader(root, max_examples=25)
        self.bundle = self.loader.load()

    def test_proxy_scores_flagged_as_gpt41(self) -> None:
        sources = {entry.get("human_score_source") for entry in self.bundle.entries}
        self.assertIn("proxy_gpt4.1_score", sources)
        providers = {
            entry.get("human_score_provider")
            for entry in self.bundle.entries
            if entry.get("human_score_source") == "proxy_gpt4.1_score"
        }
        self.assertSetEqual(providers, {"gpt-4.1"})

    def test_sample_scores_are_all_five(self) -> None:
        eval_path = (
            pointwise_pipeline.PROJECT_ROOT
            / "prometheus-eval"
            / "BiGGen-Bench"
            / "sample_evals.json"
        )
        with eval_path.open() as f:
            payload = json.load(f)
        score_values = {record.get("score") for record in payload.values() if "score" in record}
        self.assertEqual(score_values, {5})


class PromptPerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mixin = SimplePromptPerturbationMixin()
        self.mixin.base = types.SimpleNamespace(
            prompt_processor=types.SimpleNamespace(
                SYSTEM_INSTRUCTIONS="You are a meticulous teaching assistant."
            )
        )

    def _build_example(self) -> PromptExample:
        sections = {
            "assignment": "one two three four five six seven eight nine ten",
            "rubric": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "instructor_criteria": "Provide detailed justification for each step.",
            "reference_answer": "ref one two three four five six seven eight nine",
            "reference_materials": [
                {"relative_path": "guide.txt", "content": "line1 line2 line3 line4 line5"}
            ],
        }
        metadata = {
            "question_id": "q0",
            "max_score": 42,
            "human_scores": [10, 20, 30],
            "context_files": sections["reference_materials"],
            "sections": sections,
            "student_answer": "student response with multiple tokens for perturbation",
        }
        return PromptExample(
            id="q0-1",
            instruction="placeholder instruction",
            output="student response with multiple tokens for perturbation",
            metadata=metadata,
        )

    def test_variants_are_distinct(self) -> None:
        example = self._build_example()
        variants, info = self.mixin._build_variants([example])

        self.assertIn("original", variants)
        self.assertIn("perturbed", variants)

        original_example = variants["original"][0]
        perturbed_example = variants["perturbed"][0]

        self.assertNotEqual(original_example.instruction, perturbed_example.instruction)
        self.assertNotEqual(original_example.output, perturbed_example.output)

        self.assertIn("apple", perturbed_example.instruction)
        self.assertIn("capricorn", perturbed_example.output)

        orig_sections = original_example.metadata["sections"]
        pert_sections = perturbed_example.metadata["sections"]

        self.assertNotEqual(orig_sections["rubric"], pert_sections["rubric"])
        self.assertNotEqual(orig_sections["reference_answer"], pert_sections["reference_answer"])
        self.assertEqual(orig_sections["instructor_criteria"], pert_sections.get("instructor_criteria"))

        perturb_list = info[example.id]
        labels = {entry.get("label") for entry in perturb_list}
        self.assertSetEqual(labels, {"original", "perturbed"})
        original_info = next(entry for entry in perturb_list if entry.get("label") == "original")
        perturbed_info = next(entry for entry in perturb_list if entry.get("label") == "perturbed")
        self.assertIn("student_answer", perturbed_info)
        self.assertNotEqual(
            original_info.get("student_answer"),
            perturbed_info.get("student_answer"),
        )


class HelperFunctionTests(unittest.TestCase):
    def test_simple_parser(self) -> None:
        record = {"response": [{"text": "17"}]}
        result, fail = simple_pointwise_parse(record)
        self.assertIsInstance(result, PointwiseScore)
        self.assertEqual(result.score, 17)
        self.assertFalse(fail)

        tagged_record = {"response": [{"text": "\n<score> 3 </score>\n"}]}
        result, fail = simple_pointwise_parse(tagged_record)
        self.assertEqual(result.score, 3)
        self.assertFalse(fail)

        bad_record = {"response": [{"text": "invalid"}]}
        result, fail = simple_pointwise_parse(bad_record)
        self.assertTrue(fail)
        self.assertEqual(result.score, 0)

    def test_result_evaluator(self) -> None:
        evaluator = PointwiseResultEvaluator()
        human = [1, 2, 3]
        predictions = [
            JudgePrediction.from_dict({
                "id": "a",
                "prompt": [],
                "response": [],
                "metadata": {},
                "result": {"score": 1},
            }),
            JudgePrediction.from_dict({
                "id": "b",
                "prompt": [],
                "response": [],
                "metadata": {},
                "result": {"score": 2},
            }),
            JudgePrediction.from_dict({
                "id": "c",
                "prompt": [],
                "response": [],
                "metadata": {},
                "result": {"score": 3},
            }),
        ]
        summary = evaluator.evaluate(human, predictions)
        self.assertAlmostEqual(summary.pearson_corr, 1.0)


class MetricsAnalysisTests(unittest.TestCase):
    def test_var_std_under_perturbation(self) -> None:
        stats = var_std_under_perturbation([1, 2, 3, 4])
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["variance"], 1.6666666, places=5)
        self.assertEqual(stats["n"], 4)

    def test_analyze_pointwise_results_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result_path = tmp_path / "sample.jsonl"
            meta_path = tmp_path / "sample.meta.json"

            record_one = {
                "id": "ex1",
                "metadata": {
                    "max_score": 5,
                    "human_scores": [4, 4, 5],
                    "perturbation_scores": {"original": 4.5, "perturbed": 3.5},
                    "perturbations": [
                        {"label": "original", "score": 4.5},
                        {"label": "perturbed", "score": 3.5},
                    ],
                },
                "prompt": [],
                "response": [],
                "result": {"score": 4.5},
            }
            record_two = {
                "id": "ex2",
                "metadata": {
                    "max_score": 5,
                    "human_scores": [],
                    "perturbation_scores": {"original": 2.0},
                    "perturbations": [{"label": "original", "score": 2.0}],
                },
                "prompt": [],
                "response": [],
                "result": {"score": 2.0},
            }

            with result_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(record_one) + "\n")
                handle.write(json.dumps(record_two) + "\n")

            meta_payload = {
                "pearson_corr": 0.5,
                "model_scores": [4.5, 2.0],
                "human_means": [4.333333, 0.0],
                "per_example_stats": [],
            }
            meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")

            sections = analyze_pointwise_results(
                result_path,
                meta_path=meta_path,
                include_meta=True,
            )

            summary = sections.get("summary", {})
            self.assertEqual(summary.get("examples_in_file"), 2)
            self.assertEqual(summary.get("examples_with_perturbations"), 1)
            self.assertAlmostEqual(summary.get("meta_pearson_corr"), 0.5)

            dispersion = sections.get("perturbation_dispersion", {})
            self.assertEqual(dispersion.get("total_perturbed_scores"), 1)

    def test_continuous_robustness_scaling(self) -> None:
        metrics = continuous_robustness([3, 5, 4], 4, scale_max=5)
        self.assertGreater(metrics["MAD_O"], 0.0)
        self.assertIn("MAD_O_norm", metrics)


if __name__ == "__main__":
    unittest.main()
