# Prompt Perturbation – Agent Notes

## Environment
- Primary setup is on the cluster; always run `module load miniconda` first.
- Prefer the `senior_project_research` Conda env: `module load miniconda && conda run -n senior_project_research python …`.
- If jobs are killed (OOM or walltime), retry with Slurm wrappers, e.g.  
  `srun --mem=64G --time=1:00:00 -p gpu_devel --gpus=2 "module load miniconda && conda run -n senior_project_research python pointwise_pipeline.py --overwrite_ok=true"` for quick checks; switch to `-p gpu` for longer runs.
- The `gpu` partition supports 48-hour jobs and enforces a per-user cap of 24 GPUs; pick it when submitting longer `sbatch` runs.
- `run_models.py` now launches the 8B judge with `--tensor_parallel_size 2`, so request at least two GPUs in the corresponding `sbatch` submission.
- The pipeline exports `VLLM_USE_CUDA_GRAPH=0`, `VLLM_USE_FLEX_ATTENTION=0`, and `VLLM_WORKER_WARMUP_NUM_REQUESTS=128` automatically; no need to set them manually.
- FlashAttention 2 is unsupported on V100 nodes; either pin Slurm jobs to A100-80G with `--constraint=a100-80g` or export `VLLM_USE_FA2=0` and `VLLM_USE_FLASH_ATTENTION=0` before launching vLLM runs to avoid engine startup failures.
- Metrics tooling now also reports MAE/RMSE, exact-match accuracy, score-distribution similarity (JS divergence, TV distance), and judge-consistency stats alongside Pearson/Spearman/Kappa.
- Dataset outputs are organized by system-prompt provenance: datasets using our default prompts live in `results/default_system_prompts/<dataset>/`, while those with embedded prompts (e.g., BiGGen Bench) live in `results/dataset_system_prompts/<dataset>/` (symlinks keep the legacy paths working).
- Pointwise runs now enforce a minimum result coverage ratio (default 0.95). If fewer than 95 % of examples yield parsed scores, the pipeline raises an error instead of silently writing partial files.
- A validation helper is available at `scripts/validate_results.py` to confirm coverage after a run. Example:
  - `module load miniconda && conda run -n senior_project_research python scripts/validate_results.py --dataset llm_grader --results results/default_system_prompts/llm_grader/llm_grader_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct.jsonl`
  - By default the script requires at least 95 % coverage; override with `--min-ratio` or `--expected-count` for smoke tests.
- Recent sanity run (2 items) succeeded with `--mem=128G` on partition `gpu` (job 46798649); treat 128G as the safe baseline for dual-judge jobs.
- Llama BigGen Bench sweeps must request `--constraint=a100-80g` to keep VLLM stable; use the standard flags (`--gpu_memory_utilization 0.60 --swap_space 8 --max_input_len 1024 --max_model_len 1152 --enforce_eager true --max_num_seqs 64`) and include `--overwrite_ok=true` after cleanup. Qwen runs can stay on the general `gpu` pool without extra constraints unless debugging hardware issues.
- Ensure `results/biggen_bench` is a writable directory before launching Qwen BigGen jobs; if a legacy symlink exists, remove it so the pipeline can recreate the directory (it is safe to add the symlink back after runs complete if needed).
- Likewise for the default datasets (`results/llm_grader`, `results/flask`, `results/mt_bench`, `results/chatbot_arena`), replace legacy symlinks with real directories before multi-dataset runs; restore the links afterward if downstream tooling relies on them.
- Larger Qwen runs (BigGen paraphrase counts ≥5 or multi-dataset sweeps) require A100-80G nodes to avoid KV-cache OOM on 16 GB GPUs—use `--constraint=a100-80g` for those, while smaller N (≤3) can stay on the general `gpu` pool.
- When working from the local machine, reconnect via `ssh <host>` before running commands; if the connection fails due to a lingering control socket, rerun the SSH command with escalation (which clears the socket) and proceed.
- Bouchet cluster GPU notes:

## BiGGen-Bench score blacklist
The judge occasionally copies raw integers (action IDs, timestamp offsets, plan lengths, etc.) into the score channel. When that happens the deviations blow past the 0–5 rubric and corrupt robustness metrics. Until upstream fixes the grading prompts, `BiGGenDatasetLoader` automatically skips the following tasks (examples reference the latest qwen2.5-14B outputs):

- `api_documentation` (tool_usage) – perturbation:original=-10.0 in example `tool_usage_api_documentation_1` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:627)
- `checklist_generation` (theory_of_mind) – perturbation:original=7.0 in example `theory_of_mind_checklist_generation_4` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:530)
- `code_revision` (refinement) – perturbation:original=27.0 in example `refinement_code_revision_7` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:378)
- `coding_for_math` (tool_usage) – perturbation:original=8.0 in example `tool_usage_coding_for_math_5` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:641)
- `competition_mwp` (reasoning) – result_score=529.0 in example `reasoning_competition_mwp_0` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:281)
- `compositional_planning` (planning) – perturbation:original=6.0 in example `planning_compositional_planning_1` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:202)
- `deductive` (reasoning) – perturbation:paraphrase_14=-2.0 in example `reasoning_deductive_1` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp1p0.jsonl:292)
- `determine_what_is_wrong` (safety) – perturbation:original=10.0 in example `safety_determine_what_is_wrong_6` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:453)
- `essay_revision` (refinement) – perturbation:original=13.0 in example `refinement_essay_revision_1` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:382)
- `executable_actions` (instruction_following) – perturbation:paraphrase_10=10.0 in example `instruction_following_executable_actions_9` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp1p0.jsonl:140)
- `faithful_explanation` (instruction_following) – perturbation:paraphrase_7=7.0 in example `instruction_following_faithful_explanation_0` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p7.jsonl:141)
- `faux_pas_explanation` (theory_of_mind) – perturbation:original=6.0 in example `theory_of_mind_faux_pas_explanation_8` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:544)
- `high_school_mwp` (reasoning) – perturbation:paraphrase_2=7.5 in example `reasoning_high_school_mwp_2` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p3.jsonl:313)
- `hypothesis_proposal` (reasoning) – perturbation:paraphrase_9=110.0 in example `reasoning_hypothesis_proposal_5` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp1p0.jsonl:326)
- `inductive` (reasoning) – perturbation:paraphrase_6=-1.0 in example `reasoning_inductive_1` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p7.jsonl:332)
- `json_csv_xml` (grounding) – result_score=100.0 in example `grounding_json_csv_xml_0` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:31)
- `knowledge_graph` (theory_of_mind) – perturbation:paraphrase_16=6.0 in example `theory_of_mind_knowledge_graph_9` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p3.jsonl:575)
- `legal_reason` (reasoning) – perturbation:paraphrase_2=198.0 in example `reasoning_legal_reason_4` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp1p0.jsonl:345)
- `math_proof` (reasoning) – result_score=90.0 in example `reasoning_math_proof_0` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:351)
- `mentioning_potential_harm` (safety) – perturbation:original=8.0 in example `safety_mentioning_potential_harm_4` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:501)
- `moral_belief` (safety) – perturbation:original=8.0 in example `safety_moral_belief_1` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:508)
- `multi_step` (tool_usage) – result_score=95.0 in example `tool_usage_multi_step_6` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:662)
- `multi_task_inference` (instruction_following) – perturbation:original=894.0 in example `instruction_following_multi_task_inference_4` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:185)
- `personal_assistant` (planning) – perturbation:original=7.0 in example `planning_personal_assistant_4` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:235)
- `rationale_revision` (refinement) – result_score=54.0 in example `refinement_rationale_revision_0` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:409)
- `replanning` (refinement) – result_score=8.0 in example `refinement_replanning_8` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:426)
- `response_generation` (theory_of_mind) – perturbation:original=7.0 in example `theory_of_mind_response_generation_8` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p7.jsonl:594)
- `revision_with_tools` (refinement) – result_score=7.0 in example `refinement_revision_with_tools_2` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:430)
- `reward_modeling` (planning) – perturbation:original=-1.0 in example `planning_reward_modeling_9` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:250)
- `search_engine` (tool_usage) – perturbation:paraphrase_20=2024.0 in example `tool_usage_search_engine_3` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p7.jsonl:669)
- `self_correction` (refinement) – result_score=10.0 in example `refinement_self_correction_2` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:440)
- `table_reason` (reasoning) – perturbation:original=7.0 in example `reasoning_table_reason_0` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:361)
- `temporal_grounding` (grounding) – perturbation:original=2010.0 in example `grounding_temporal_grounding_6` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:97)
- `time_traveler_dilemma` (theory_of_mind) – perturbation:paraphrase_18=16.0 in example `theory_of_mind_time_traveler_dilemma_0` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp1p0.jsonl:606)
- `tool_making` (tool_usage) – perturbation:paraphrase_14=100.0 in example `tool_usage_tool_making_2` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp1p0.jsonl:678)
- `web_browsing` (tool_usage) – result_score=4017.0 in example `tool_usage_web_browsing_2` (results/biggen_bench/biggen_bench_pointwise_vanilla.base_pointwise.qwen2.5-14b-instruct-temp0p0.jsonl:688)


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

## Grace Cluster: GPU Partitions and Smoke Testing

- Partitions overview (Grace)
  - gpu
    - a5000: 11 nodes, 4×24GB VRAM, 206GB RAM/node
    - a100-80g: 3 nodes, 4×80GB VRAM, 984GB RAM/node (2×CPU 6342, 1×CPU 6326)
    - a100-40g: 2 nodes, 4×40GB VRAM, 361GB RAM/node
    - v100: 4 nodes, 4×16GB VRAM, 370GB RAM/node; 2 nodes, 2×16GB VRAM, 90GB RAM/node
    - rtx2080ti: 3 nodes, 4×11GB VRAM, 181GB RAM/node
    - rtx5000: 2 nodes, 4×16GB VRAM, 181GB RAM/node
  - gpu_devel
    - a100-80g: 1 node, 4×80GB VRAM, 984GB RAM/node
    - a5000: 1 node, 4×24GB VRAM, 206GB RAM/node
    - rtx5000: 4 nodes, 4×16GB VRAM, 181GB RAM/node
    - v100: 1 node, 4×16GB VRAM, 370GB RAM/node
    - rtx2080ti: 2 nodes, 4×11GB VRAM, 181GB RAM/node
    - rtx3090: 2 nodes, 4×24GB VRAM, 166GB RAM/node

- Scheduling guidance
  - Use partition `gpu` for longer jobs (up to 48h). The `gpu` partition enforces a per‑user cap of 24 GPUs.
  - Pin to a single host to avoid Ray/vLLM placement issues: include `--nodes=1 --gpus-per-node=2`.
  - The 8B judge is run with `--tensor_parallel_size 2`; request at least two GPUs in `sbatch`/`srun`.
  - Memory baselines observed: 128G RAM on `gpu` is a safe baseline for dual‑judge jobs; for smoke tests on `gpu_devel`, 32G RAM is typically sufficient for system memory (GPU VRAM is the limiting factor).

- Environment conventions
  - Always prefix with: `module load miniconda && conda run -n senior_project_research <command>`.
  - Avoid `conda activate/deactivate` in batch scripts; prefer `conda run` within `--wrap` or script files.

- Submit templates
  - Smoke (both models) on `gpu_devel` (A100‑80G preferred)
    - sbatch --job-name=pp_smoke_all \
      --partition=gpu_devel \
      --nodes=1 --gpus-per-node=2 \
      --time=0:30:00 --mem=32G \
      --output=logs/run_model_%j.out \
      --wrap="module load miniconda && conda run -n senior_project_research \
              python run_models.py --smoke_test --max_items 2 --datasets llm_grader \
              --retries 1 --model_download_dir /home/vl298/.cache/huggingface --overwrite_ok"
    - Optionally add `--constraint=a100-80g` to prefer A100‑80G.
- If logs show “No available memory for the cache blocks”: increase reserved VRAM or shrink context
    - Add to the wrap: `--extra_pipeline_args --max_input_len 1024 --max_model_len 1152 --gpu_memory_utilization 0.75`
- If warmup CUDA OOM occurs during model load: lower utilization and/or context
    - Add to the wrap: `--extra_pipeline_args --gpu_memory_utilization 0.60 --swap_space 8`
- Current best-known stable flags for vLLM (both models) to avoid CUDA graph / flex attention allocator crashes:
  - Append `--extra_pipeline_args --max_input_len 1024 --max_model_len 1152 --gpu_memory_utilization 0.60 --swap_space 8 --enforce_eager true --max_num_seqs 64`
- Qwen-only smoke (helpful when Llama succeeds but Qwen fails):
  - `sbatch --job-name=pp_smoke_qwen --partition=gpu_devel --nodes=1 --gpus-per-node=2 --constraint=a100-80g --time=0:30:00 --mem=32G --output=logs/run_model_%j.out --wrap="module load miniconda && conda run -n senior_project_research python pointwise_pipeline.py --use_dummy_model=false --model_backend=hfvllm --model_pt=Qwen/Qwen2.5-14B-Instruct --model_name qwen2.5-14b-instruct --datasets llm_grader --max_items 2 --tensor_parallel_size 2 --max_input_len 1024 --max_model_len 1152 --gpu_memory_utilization 0.60 --swap_space 8 --model_download_dir /home/vl298/.cache/huggingface --overwrite_ok=true --disable_prompt_perturbation --enforce_eager true"`
  - Full 48h run on `gpu`, single node, 2 GPUs, 128G RAM
    - sbatch --job-name=pp_full \
      --partition=gpu \
      --nodes=1 --gpus-per-node=2 \
      --time=48:00:00 --mem=128G \
      --output=logs/run_model_%j.out \
      --wrap="module load miniconda && conda run -n senior_project_research \
              python run_models.py --max_items 10000 --datasets llm_grader \
              --model_download_dir /home/vl298/.cache/huggingface --overwrite_ok"

- Quick monitoring
  - sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
  - tail -f logs/run_model_<JOBID>.out
- Post-run validation
  - After a job finishes, confirm coverage with `scripts/validate_results.py` (see example above). Treat a run as unsuccessful if the validator reports coverage below threshold or exits non-zero.

- vLLM notes on mixed GPUs
  - RTX 2080 Ti (compute 7.5) lacks BF16 and FA2 support; vLLM falls back automatically (warnings expected).
  - “No available memory for the cache blocks” → increase `--gpu_memory_utilization` (e.g., 0.70–0.80) or reduce `--max_model_len/--max_input_len`.
  - Warmup OOM during model load → decrease `--gpu_memory_utilization` (e.g., 0.55–0.65), reduce context, and/or set `--swap_space 8`.
  - Keep `--tensor_parallel_size 2` and `--nodes=1` for 8B judge stability.
