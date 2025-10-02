"""Object-oriented pointwise evaluation pipeline for the llm_grader dataset."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

from evaluation_interfaces import PointwiseEval
from pipelines import PointwisePipeline


def configure_logging(args: argparse.Namespace) -> Path:
    """Configure logging and return the path to the log file."""

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    flag_pairs = [f"{key}-{getattr(args, key)}" for key in sorted(vars(args))]
    flag_str = "_".join(flag_pairs)
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
    pipeline = PointwisePipeline(
        model=model,
        eval_callable=eval_callable,
        reife_root=reife_root,
        model_name=model_name,
        enable_prompt_perturbation=not args.disable_prompt_perturbation,
    )
    pipeline.run(overwrite_ok=args.overwrite_ok)


if __name__ == "__main__":
    main()
