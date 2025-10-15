"""Object-oriented pointwise evaluation pipeline for the llm_grader dataset."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

from evaluation_interfaces import PointwiseEval
from data_loaders import (
    BiGGenDatasetLoader,
    ChatbotArenaLoader,
    DatasetLoader,
    FLASKDatasetLoader,
    LLMGraderDatasetLoader,
    MTBenchHumanJudgmentsLoader,
)
from pipelines import PointwisePipeline


def configure_logging(args: argparse.Namespace) -> Path:
    """Configure logging and return the path to the log file."""

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize(value: object) -> str:
        if isinstance(value, Path):
            value = value.as_posix()
        text = str(value)
        text = text.replace(os.sep, "_")
        if os.path.altsep:
            text = text.replace(os.path.altsep, "_")
        text = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
        return text[:24]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    flag_pairs = [f"{key}-{_sanitize(getattr(args, key))}" for key in sorted(vars(args))]
    raw_flag_str = "_".join(flag_pairs)
    if len(raw_flag_str) > 200:
        prefix = "_".join(flag_pairs[:5])
        digest = hashlib.sha256(raw_flag_str.encode("utf-8")).hexdigest()[:12]
        flag_str = f"{prefix}_hash-{digest}"
    else:
        flag_str = raw_flag_str
    log_path = logs_dir / f"pointwise_pipeline_{timestamp}_{flag_str}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s %(message)s")
    )

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return log_path


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_dummy_model() -> Tuple[object, str]:
    from importlib import util as importlib_util

    try:
        from tests.dummy_pointwise_model import DummyPointwiseAPI as _API  # type: ignore
    except ModuleNotFoundError:
        module_path = PROJECT_ROOT / "ReIFE" / "tests" / "dummy_pointwise_model.py"
        spec = importlib_util.spec_from_file_location("dummy_pointwise_model", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _API = module.DummyPointwiseAPI
    model = _API(
        model_pt="dummy",
        parallel_size=1,
        initial_wait_time=0,
        end_wait_time=0,
        max_retries=1,
    )
    return model, "dummy_api"


def load_real_model() -> Tuple[object, str]:
    from ReIFE.models import get_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Cannot run Qwen model without GPU.")

    model_cls = get_model("hfvllm")
    home_dir = os.getenv("HOME", "")
    model = model_cls(
        model_pt="Qwen/Qwen2.5-0.5B-Instruct",
        tensor_parallel_size=1,
        download_dir=os.path.join(home_dir, ".cache/huggingface/hub"),
        gpu_memory_utilization=0.5,
        quantization=None,
        swap_space=2,
        max_input_len=512,
        max_model_len=768,
        dtype="auto",
    )
    return model, "qwen2.5-0.5b"


def create_model(use_dummy: bool) -> Tuple[object, str]:
    if use_dummy:
        return load_dummy_model()
    return load_real_model()


def log_callable_metadata(model_name: str, eval_callable: PointwiseEval) -> None:
    import inspect

    target = eval_callable._eval_fn  # type: ignore[attr-defined]
    logging.info("Using model: %s", model_name)
    logging.debug("Evaluator target: %s", target)
    logging.debug("Signature: %s", inspect.signature(target))
    doc = inspect.getdoc(target) or "<no docstring>"
    logging.debug("Docstring:\n%s", doc)


def build_dataset_loaders(
    args: argparse.Namespace,
    *,
    reife_root: Path,
    requested: Optional[Sequence[str]] = None,
) -> dict[str, DatasetLoader]:
    """Construct dataset loader instances based on CLI arguments."""

    max_examples = args.max_items
    project_root = PROJECT_ROOT
    loaders: dict[str, DatasetLoader] = {}

    requested_set = {name for name in requested} if requested else None

    def _should_include(key: str) -> bool:
        return requested_set is None or key in requested_set

    def _register(key: str, factory) -> None:
        if not _should_include(key):
            return
        try:
            loaders[key] = factory()
        except FileNotFoundError as exc:
            logging.error("Missing required files for dataset '%s': %s", key, exc)
            raise SystemExit(1) from exc
        except Exception as exc:  # pragma: no cover - defensive path
            logging.exception("Failed to initialise dataset loader '%s'", key)
            raise SystemExit(1) from exc

    llm_grader_root = reife_root.parent / "llm_grader" / "dataset_os"
    _register(
        "llm_grader",
        lambda: LLMGraderDatasetLoader(llm_grader_root, max_examples=max_examples),
    )

    biggen_root = project_root / "prometheus-eval" / "BiGGen-Bench"
    _register(
        "biggen_bench",
        lambda: BiGGenDatasetLoader(biggen_root, max_examples=max_examples),
    )

    flask_root = project_root / "FLASK"
    flask_eval_path = Path(args.flask_eval_path)
    flask_eval_hard_path = Path(args.flask_eval_hard_path)
    flask_response_path = Path(args.flask_response_path) if args.flask_response_path else None
    flask_review_path = Path(args.flask_review_path) if args.flask_review_path else None
    _register(
        "flask",
        lambda: FLASKDatasetLoader(
            flask_root,
            split=args.flask_split,
            eval_path=flask_eval_path,
            hard_path=flask_eval_hard_path,
            response_path=flask_response_path,
            review_path=flask_review_path,
            max_examples=max_examples,
        ),
    )

    hf_chunk_pct = args.hf_chunk_pct if args.hf_chunk_pct is not None else 10.0

    _register(
        "mt_bench",
        lambda: MTBenchHumanJudgmentsLoader(
            project_root,
            data_dir=args.mtb_data_dir,
            max_examples=max_examples,
            scoring_scale=args.scoring_scale,
            hf_chunk_pct=hf_chunk_pct,
        ),
    )

    _register(
        "chatbot_arena",
        lambda: ChatbotArenaLoader(
            project_root,
            data_dir=args.arena_data_dir,
            max_examples=max_examples,
            scoring_scale=args.scoring_scale,
            hf_chunk_pct=hf_chunk_pct,
        ),
    )

    return loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pointwise evaluation demo")
    parser.add_argument(
        "--use_dummy_model",
        type=str2bool,
        default=True,
        help="Whether to use the lightweight dummy model (default: true).",
    )
    parser.add_argument(
        "--overwrite_ok",
        type=str2bool,
        default=False,
        help="Allow overwriting existing output_text and meta-eval files.",
    )
    parser.add_argument(
        "--disable_prompt_perturbation",
        action="store_true",
        help="Disable prompt perturbation mixin (default: enabled).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of dataset keys to evaluate. Supported keys: llm_grader, biggen_bench, "
            "flask, mt_bench, chatbot_arena. "
            "Defaults to running all datasets."
        ),
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Limit the number of evaluation examples per dataset (useful for tests).",
    )
    parser.add_argument(
        "--flask_split",
        type=str,
        default="standard",
        choices=["standard", "hard"],
        help="Which FLASK split to evaluate (standard or hard).",
    )
    parser.add_argument(
        "--flask_eval_path",
        type=str,
        default=str(PROJECT_ROOT / "FLASK" / "evaluation_set" / "flask_evaluation.jsonl"),
        help="Path to the FLASK standard evaluation jsonl file.",
    )
    parser.add_argument(
        "--flask_eval_hard_path",
        type=str,
        default=str(PROJECT_ROOT / "FLASK" / "evaluation_set" / "flask_hard_evaluation.jsonl"),
        help="Path to the FLASK hard evaluation jsonl file.",
    )
    parser.add_argument(
        "--flask_response_path",
        type=str,
        default=str(PROJECT_ROOT / "FLASK" / "model_output" / "outputs" / "gpt4.jsonl"),
        help="Optional path to pre-generated FLASK model responses.",
    )
    parser.add_argument(
        "--flask_review_path",
        type=str,
        default=str(PROJECT_ROOT / "FLASK" / "gpt_review" / "outputs" / "gpt4_review.jsonl"),
        help="Optional path to GPT-based FLASK review scores used as human reference.",
    )
    parser.add_argument(
        "--mtb_data_dir",
        type=str,
        default=None,
        help="Optional data directory for the MT-Bench human judgments dataset (Hugging Face snapshot).",
    )
    parser.add_argument(
        "--arena_data_dir",
        type=str,
        default=None,
        help="Optional data directory for the Chatbot Arena conversations dataset (Hugging Face snapshot).",
    )
    parser.add_argument(
        "--hf_chunk_pct",
        type=float,
        default=10.0,
        help=(
            "When loading Hugging Face datasets (MT-Bench, Chatbot Arena), load in this percentage-sized chunks "
            "(default: 10). Set to 0 or a value >=100 to load the full dataset."
        ),
    )
    parser.add_argument(
        "--pairwise_mode",
        type=str2bool,
        default=False,
        help="Enable pairwise judge mode for MT-Bench and Chatbot Arena (optional).",
    )
    parser.add_argument(
        "--scoring_scale",
        type=int,
        default=10,
        choices=[5, 10],
        help="Target scoring scale for pairwise datasets when normalizing scores.",
    )
    args = parser.parse_args()

    log_path = configure_logging(args)
    logging.info("Logging to %s", log_path)
    logging.info(
        "Prompt perturbation enabled: %s", not args.disable_prompt_perturbation
    )

    model, model_name = create_model(args.use_dummy_model)
    eval_callable = PointwiseEval()
    log_callable_metadata(model_name, eval_callable)

    reife_root = PROJECT_ROOT / "ReIFE"
    dataset_names = None
    if args.datasets:
        dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    if args.pairwise_mode:
        logging.warning(
            "Pairwise judge mode is not implemented yet; continuing with numeric rescoring only."
        )

    if args.max_items:
        logging.info("Limiting each dataset to the first %d examples", args.max_items)

    pipeline = PointwisePipeline(
        model=model,
        eval_callable=eval_callable,
        reife_root=reife_root,
        model_name=model_name,
        dataset_loaders=build_dataset_loaders(
            args,
            reife_root=reife_root,
            requested=dataset_names,
        ),
        datasets=dataset_names,
        enable_prompt_perturbation=not args.disable_prompt_perturbation,
        results_root=PROJECT_ROOT / "results",
        logs_root=PROJECT_ROOT / "logs",
    )
    if dataset_names:
        logging.info("Restricting evaluation to datasets: %s", ", ".join(dataset_names))
    pipeline.run(overwrite_ok=args.overwrite_ok)
    logging.info("Pipeline run completed")


if __name__ == "__main__":
    main()
