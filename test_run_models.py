"""Tests for run_models orchestration helper."""

from __future__ import annotations

import tempfile
import types
import unittest

import run_models


class RunModelsTests(unittest.TestCase):
    def _base_namespace(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            python="python",
            max_items=3,
            datasets=None,
            overwrite_ok=False,
            model_download_dir=None,
            gpt_key_path=None,
            gpt_account_path=None,
            extra_pipeline_args=None,
            dry_run=False,
            retries=5,
        )

    def test_build_command_requires_api_credentials(self) -> None:
        spec = run_models.ModelSpec(
            name="api-model",
            backend="gpt",
            model_pt="gpt-oss",
            requires_api_key=True,
            requires_account=True,
        )
        args = self._base_namespace()
        cmd = run_models.build_command(spec, args)
        self.assertEqual(cmd, [])

    def test_build_command_includes_download_dir(self) -> None:
        spec = run_models.ModelSpec(
            name="qwen",
            backend="hfvllm",
            model_pt="Qwen/Qwen3-7B-Instruct",
            extra_args=("--model_name", "qwen3-7b-instruct"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._base_namespace()
            args.model_download_dir = tmpdir
            cmd = run_models.build_command(spec, args)
        self.assertIn(f"--model_download_dir={tmpdir}", cmd)
        self.assertIn("--model_name", cmd)
        self.assertIn("qwen3-7b-instruct", cmd)

    def test_build_command_accepts_extra_args(self) -> None:
        spec = run_models.ModelSpec(
            name="prometheus",
            backend="prometheusvllm",
            model_pt="kaist-ai/prometheus-7b-v2.0",
        )
        args = self._base_namespace()
        args.extra_pipeline_args = ["--datasets=llm_grader", "--overwrite_ok=true"]
        cmd = run_models.build_command(spec, args)
        self.assertTrue(cmd[-2].startswith("--datasets="))
        self.assertEqual(cmd[-1], "--overwrite_ok=true")


if __name__ == "__main__":
    unittest.main()
