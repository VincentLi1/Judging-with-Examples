#!/usr/bin/env python3
"""Run pointwise_pipeline.py across the configured set of judge models."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = REPO_ROOT / "pointwise_pipeline.py"


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a single model invocation."""

    name: str
    backend: str
    model_pt: str
    extra_args: Sequence[str] = field(default_factory=tuple)
    requires_api_key: bool = False
    requires_account: bool = False


MODEL_SPECS: Sequence[ModelSpec] = (
    ModelSpec(
        name="llama-3.1-8b-instruct",
        backend="hfvllm",
        model_pt="meta-llama/Meta-Llama-3.1-8B-Instruct",
        extra_args=(
            "--model_name",
            "llama-3.1-8b-instruct",
            "--tensor_parallel_size",
            "1",
            "--gpu_memory_utilization",
            "0.85",
            "--max_input_len",
            "3584",
            "--max_model_len",
            "4096",
        ),
    ),
    ModelSpec(
        name="qwen25-14b-instruct",
        backend="hfvllm",
        model_pt="Qwen/Qwen2.5-14B-Instruct",
        extra_args=(
            "--model_name",
            "qwen2.5-14b-instruct",
            "--tensor_parallel_size",
            "2",
            "--gpu_memory_utilization",
            "0.9",
            "--swap_space",
            "8",
            "--max_input_len",
            "3584",
            "--max_model_len",
            "4096",
        ),
    ),
)




def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pointwise pipeline for all configured judge models."
    )
    parser.add_argument(
        "--max_items",
        type=int,
        required=True,
        help="Limit the number of examples per dataset for each run.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset list to forward to the pipeline.",
    )
    parser.add_argument(
        "--overwrite_ok",
        action="store_true",
        help="Pass --overwrite_ok=true to pointwise_pipeline (defaults to false).",
    )
    parser.add_argument(
        "--model_download_dir",
        type=str,
        default=None,
        help="Shared cache directory for Hugging Face downloads (recommended for clusters).",
    )
    parser.add_argument(
        "--gpt_key_path",
        type=str,
        default=None,
        help="Path to API key file for GPT-OSS backend.",
    )
    parser.add_argument(
        "--gpt_account_path",
        type=str,
        default=None,
        help="Path to account / organisation file for GPT-OSS backend.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable or "python",
        help="Python executable used to launch pointwise_pipeline.py.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Maximum number of attempts per model command before giving up.",
    )
    parser.add_argument(
        "--extra_pipeline_args",
        type=str,
        nargs=argparse.REMAINDER,
        help="Additional arguments appended to every pointwise_pipeline invocation.",
    )
    return parser.parse_args(argv)


def ensure_path(path_str: str | None, description: str) -> str:
    if not path_str:
        raise FileNotFoundError(f"{description} not provided; cannot run API-backed model.")
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    return str(path)


def build_command(
    spec: ModelSpec,
    args: argparse.Namespace,
    *,
    force_overwrite: bool = False,
) -> list[str]:
    cmd: list[str] = [
        args.python,
        str(PIPELINE_PATH),
        "--use_dummy_model=false",
        f"--model_backend={spec.backend}",
        f"--model_pt={spec.model_pt}",
        f"--max_items={args.max_items}",
    ]

    if args.datasets:
        cmd.append(f"--datasets={args.datasets}")

    if args.model_download_dir:
        cmd.append(f"--model_download_dir={args.model_download_dir}")

    cmd.extend(spec.extra_args)

    overwrite_enabled = bool(getattr(args, "overwrite_ok", False) or force_overwrite)
    if overwrite_enabled:
        cmd.append("--overwrite_ok=true")

    if spec.requires_api_key:
        if not getattr(args, "gpt_key_path", None):
            print(f'Skipping {spec.name} (missing API key)')
            return []
        key_path = ensure_path(args.gpt_key_path, 'GPT API key file')
        cmd.append(f'--api_key_path={key_path}')
    if spec.requires_account:
        if not getattr(args, "gpt_account_path", None):
            print(f'Skipping {spec.name} (missing API account)')
            return []
        account_path = ensure_path(args.gpt_account_path, 'GPT API account file')
        cmd.append(f'--api_account_path={account_path}')

    if args.extra_pipeline_args:
        cmd.extend(args.extra_pipeline_args)

    return cmd




def run_all_models(cli_args: argparse.Namespace) -> None:
    failures: list[tuple[str, int]] = []
    max_attempts = max(1, getattr(cli_args, "retries", 1))

    for spec in MODEL_SPECS:
        success = False
        skipped = False
        last_code = 0

        for attempt in range(1, max_attempts + 1):
            force_overwrite = bool(getattr(cli_args, "overwrite_ok", False) or attempt > 1)
            command = build_command(spec, cli_args, force_overwrite=force_overwrite)
            if not command:
                skipped = True
                break

            printable = " ".join(shlex.quote(part) for part in command)
            print(f"\n=== Running {spec.name} (attempt {attempt}/{max_attempts}) ===")
            print(f"$ {printable}")

            if cli_args.dry_run:
                success = True
                break

            result = subprocess.run(command, check=False)
            last_code = result.returncode
            if result.returncode == 0:
                success = True
                if attempt > 1:
                    print(f"[INFO] Model '{spec.name}' succeeded on attempt {attempt}.")
                break

            if attempt < max_attempts:
                print(
                    f"[WARN] Model '{spec.name}' failed with code {result.returncode}; retrying ({attempt + 1}/{max_attempts})"
                )
            else:
                print(
                    f"[ERROR] Model '{spec.name}' failed after {max_attempts} attempts (last code {result.returncode})",
                    file=sys.stderr,
                )

        if skipped:
            continue

        if cli_args.dry_run:
            continue

        if not success:
            failures.append((spec.name, last_code))
            break

    if cli_args.dry_run:
        print("\nDry run complete. Commands were not executed.")
        return

    if failures:
        failed_models = ", ".join(f"{name} (code {code})" for name, code in failures)
        raise SystemExit(f"One or more runs failed: {failed_models}")

    print("\nAll model runs completed successfully.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"pointwise_pipeline.py not found at expected path: {PIPELINE_PATH}"
        )

    run_all_models(args)


if __name__ == "__main__":
    main()
