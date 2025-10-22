# Prompt Perturbation – Agent Notes

## Environment
- Primary setup is on the cluster; always run `module load miniconda` first.
- Prefer the `senior_project_research` Conda env: `module load miniconda && conda run -n senior_project_research python …`.
- If jobs are killed (OOM or walltime), retry with Slurm wrappers, e.g.  
  `srun --mem=64G --time=1:00:00 -p gpu_devel --gpus=2 "module load miniconda && conda run -n senior_project_research python pointwise_pipeline.py --overwrite_ok=true"`.
- ReIFE submodule is vendored in `ReIFE/`; datasets cloned into `llm_grader/`, `prometheus-eval/`, `FLASK/`.

## Core Workflows
- `pointwise_pipeline.py`: runs the judge over selected datasets, optionally with prompt perturbations.  
  - Default uses a dummy model; pass `--use_dummy_model=false` to hit real models via ReIFE.
  - Important flags:  
    - Data scope: `--datasets`, `--max_items`, `--pairwise_mode`, `--scoring_scale`.  
    - FLASK sources: `--flask_split`, `--flask_eval_path`, `--flask_eval_hard_path`, `--flask_response_path`, `--flask_review_path`.  
    - HF datasets: `--mtb_data_dir`, `--arena_data_dir`, `--hf_chunk_pct`.  
    - Model config (VLLM): `--model_backend`, `--model_pt`, `--model_name`, `--tensor_parallel_size`, `--gpu_memory_utilization`, `--max_input_len`, `--max_model_len`, `--model_quantization`, `--model_download_dir`, `--model_dtype`, `--swap_space`.  
    - API config: `--api_key_path`, `--api_account_path`, `--api_parallel_size`, `--api_max_retries`, `--api_initial_wait_time`, `--api_end_wait_time`.  
    - Misc: `--overwrite_ok`, `--disable_prompt_perturbation`, `--parse_retries`.  
  - Outputs land in `results/…`; logs in `logs/pointwise_pipeline_*.log`.
- `run_models.py`: orchestrates multiple judge runs over `MODEL_SPECS`.  
  - Currently targets two VLLM judges: `meta-llama/Meta-Llama-3.1-8B-Instruct` and `Qwen/Qwen2.5-14B-Instruct`.  
  - Key flags: `--max_items` (required), `--datasets`, `--overwrite_ok`, `--model_download_dir`, `--gpt_key_path`, `--gpt_account_path`, `--python`, `--dry_run`, `--retries`, `--extra_pipeline_args`.  
  - By default API-backed judges are skipped; no key files are configured unless you pass `--gpt_key_path`/`--gpt_account_path`.  
  - Automatically forwards per-model overrides for tensor parallelism, context limits, and GPU memory ratio. Retries failed runs up to `--retries`.
- Typical invocation pattern:  
  `module load miniconda && conda run -n senior_project_research python run_models.py --max_items 10000 --model_download_dir /home/vl298/.cache/huggingface --overwrite_ok`.

## Datasets & Processing
- Supported loaders reside in `data_loaders.py`, covering LLM-Grader (`llm_grader/dataset_os`), BiGGen Bench (`prometheus-eval/BiGGen-Bench`), FLASK (`FLASK/`), MT-Bench, and Chatbot Arena (HF snapshots).  
- Prompt perturbation handled by `pipelines/pointwise.py` via an “original” plus “perturbed” variant strategy unless disabled.
- Intermediate judge reasoning and perturbation stats written alongside results (see `results/llm_grader/…` for examples).

## Testing & Validation
- Unit smoke tests: `python test_pointwise_pipeline.py` and `python test_run_models.py` (remember to wrap with `module load miniconda && conda run …`).
- `data_analysis.ipynb` provides follow-up analytics on generated scores; open after pipeline runs complete.

## Troubleshooting
- CUDA is required for VLLM backends; fallback to `--use_dummy_model=true` for quick local checks.
- Ensure API key/account files exist before enabling GPT-style backends; scripts bail early if paths are missing.
- Hugging Face cache defaults to `$HOME/.cache/huggingface/hub`; override with `--model_download_dir` to use shared storage.
