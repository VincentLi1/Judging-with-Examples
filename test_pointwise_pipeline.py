"""Unit tests for the refactored pointwise pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
import types

import numpy as np
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from evaluation_interfaces import PointwiseEval  # noqa: E402
from parsers import simple_pointwise_parse  # noqa: E402
from pipelines import PointwisePipeline, SimplePromptPerturbationMixin  # noqa: E402
from result_evaluators import PointwiseResultEvaluator  # noqa: E402
import pointwise_pipeline  # noqa: E402
from prompt_processors import PromptExample  # noqa: E402
from data_models import JudgePrediction, PointwiseScore  # noqa: E402
from metrics import analyze_pointwise_results  # noqa: E402


def _artifact_paths(pipeline: PointwisePipeline) -> tuple[Path, Path, Path]:
    results_dir = pipeline.reife_root / "results"
    outputs_dir = results_dir / "outputs"
    result_path = results_dir / pipeline._result_filename()
    output_text_path = outputs_dir / pipeline._output_text_filename()
    meta_path = results_dir / pipeline._meta_filename()
    return result_path, output_text_path, meta_path


class DummyModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model, _ = pointwise_pipeline.create_model(True)

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
        self.model, self.model_name = pointwise_pipeline.create_model(True)
        self.pipeline = PointwisePipeline(
            model=self.model,
            eval_callable=PointwiseEval(),
            reife_root=pointwise_pipeline.PROJECT_ROOT / "ReIFE",
            model_name=self.model_name,
        )
        self.result_path, self.output_text_path, self.meta_path = _artifact_paths(self.pipeline)
        for path in (self.result_path, self.output_text_path, self.meta_path):
            if path.exists():
                path.unlink()

    def tearDown(self) -> None:
        for path in (self.result_path, self.output_text_path, self.meta_path):
            if path.exists():
                path.unlink()

    def test_pipeline_run(self) -> None:
        self.pipeline.run(overwrite_ok=True)

        self.assertTrue(self.result_path.exists())
        self.assertTrue(self.output_text_path.exists())
        self.assertTrue(self.meta_path.exists())

        with self.meta_path.open() as f:
            meta = json.load(f)
        self.assertIn("pearson_corr", meta)
        self.assertTrue(np.isfinite(meta["pearson_corr"]))

        with self.result_path.open() as f:
            first = json.loads(f.readline())
        perturbations = first["metadata"].get("perturbations", [])
        self.assertIsInstance(perturbations, list)
        self.assertGreaterEqual(len(perturbations), 1)
        analysis = analyze_pointwise_results(self.result_path)
        self.assertIn("perturbation_dispersion", analysis)
        summary = analysis["perturbation_dispersion"]
        self.assertGreater(summary.get("total_perturbed_scores", 0), 0)

    def test_pipeline_run_without_perturbation(self) -> None:
        pipeline = PointwisePipeline(
            model=self.model,
            eval_callable=PointwiseEval(),
            reife_root=pointwise_pipeline.PROJECT_ROOT / "ReIFE",
            model_name=self.model_name,
            enable_prompt_perturbation=False,
        )
        result_path, output_text_path, meta_path = _artifact_paths(pipeline)
        for path in (result_path, output_text_path, meta_path):
            if path.exists():
                path.unlink()

        pipeline.run(overwrite_ok=True)

        with result_path.open() as f:
            first = json.loads(f.readline())
        self.assertIsNone(first["metadata"].get("perturbations"))

        for path in (result_path, output_text_path, meta_path):
            if path.exists():
                path.unlink()


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


if __name__ == "__main__":
    unittest.main()
