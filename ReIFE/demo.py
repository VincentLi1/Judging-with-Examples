import argparse
import json
import os

import torch

from ReIFE.evaluator import PairwiseEvaluator
from ReIFE.methods import get_method, get_parser
from ReIFE.models import get_model
from ReIFE.utils import get_dataset_path
from functools import partial
from helper import method_info, model_info, parser_info
from meta_eval import pairwise_meta_eval
from tests.dummy_model import DummyAPI


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.lower()
    if val in {"true", "t", "1", "yes", "y"}:
        return True
    if val in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def create_model(use_dummy: bool) -> tuple[object, str]:
    if use_dummy:
        model = DummyAPI(
            model_pt="dummy",
            parallel_size=1,
            initial_wait_time=0,
            end_wait_time=0,
            max_retries=1,
            key_path=None,
            account_path=None,
        )
        return model, "dummy_api"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Cannot run Qwen model without GPU.")

    model_info("hfvllm")
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


def demo(use_dummy_model: bool = True, overwrite_ok: bool = False) -> None:
    model, model_name = create_model(use_dummy_model)

    evaluator = PairwiseEvaluator(model)

    eval_method = "base_pairwise"
    method_info(eval_method)

    eval_fn = get_method(eval_method)
    eval_fn = partial(
        eval_fn,
        instruction_marker="INSTRUCTION",
        output_marker="OUTPUT",
    )

    parser_info("base_pairwise")
    parse_fn = get_parser("base_pairwise")
    parse_fn = partial(
        parse_fn,
        sys1_marker="a",
        sys2_marker="b",
        pattern=r"Output \((.*?)\)",
        verbose=True,
    )

    prompt_method = "pairwise_vanilla"
    eval_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": 64,
        "logprobs": 32,
    }

    dataset = "dummy_pairwise"
    dataset_path = get_dataset_path(dataset)
    batch_size = 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    outputs_dir = os.path.join(results_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    prompt_path = os.path.join(script_dir, "prompts", f"{prompt_method}.txt")
    result_file = os.path.join(
        results_dir,
        f"{dataset}.{model_name}.{eval_method}.{prompt_method}.jsonl",
    )
    output_text_path = os.path.join(
        outputs_dir,
        f"{dataset}.{model_name}.{eval_method}.{prompt_method}.txt",
    )
    meta_result_file = os.path.join(
        results_dir,
        f"{dataset}.{model_name}.{eval_method}.{prompt_method}.meta.json",
    )

    if not overwrite_ok:
        for path, label in (
            (output_text_path, "output_text_path"),
            (meta_result_file, "meta_result_file"),
        ):
            if os.path.exists(path):
                raise FileExistsError(
                    f"{label} '{path}' already exists. Rerun with --overwrite_ok=true to overwrite."
                )

    evaluator.pairwise_eval(
        eval_fn=eval_fn,
        input_dir=dataset_path,
        output_dir=result_file,
        prompt_dir=prompt_path,
        batch_size=batch_size,
        output_text_dir=output_text_path,
        parse_fn=parse_fn,
        verbose=True,
        **eval_kwargs,
    )
    print(f"Saving results to {result_file}")

    # Dummy heuristic flips to "Output (b)" when the prompt contains " by ",
    # giving ~0.9 accuracy and ~0.63 Krippendorff alpha on the default dataset.
    meta_result = pairwise_meta_eval(
        human=dataset_path,
        model=result_file,
        verbose=True,
    )
    with open(meta_result_file, "w", encoding="utf-8") as f:
        json.dump(meta_result, f, indent=2)
    print(f"Meta-evaluation summary: {meta_result}")
    print(f"Saving meta-eval results to {meta_result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReIFE demo evaluation")
    parser.add_argument(
        "--use_dummy_model",
        type=str2bool,
        default=True,
        help="Whether to use the dummy model (default: true). Set to false to use Qwen/Qwen2.5-0.5B-Instruct.",
    )
    parser.add_argument(
        "--overwrite_ok",
        type=str2bool,
        default=False,
        help="Allow overwriting existing output_text and meta-eval files.",
    )
    args = parser.parse_args()
    demo(use_dummy_model=args.use_dummy_model, overwrite_ok=args.overwrite_ok)
