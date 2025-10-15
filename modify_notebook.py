import json
from pathlib import Path

nb_path = Path('data_analysis.ipynb')
nb = json.loads(nb_path.read_text())

config_source = """PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / \"results\"

DATASET_CONFIG = {
    \"llm_grader\": {
        \"display\": \"LLM Grader\",
        \"result\": RESULTS_DIR / \"llm_grader\" / \"pointwise_results.jsonl\",
        \"meta\": RESULTS_DIR / \"llm_grader\" / \"summary.json\",
    },
    \"biggen_bench\": {
        \"display\": \"BiGGen-Bench\",
        \"result\": RESULTS_DIR / \"biggen_bench\" / \"pointwise_results.jsonl\",
        \"meta\": RESULTS_DIR / \"biggen_bench\" / \"summary.json\",
    },
    \"flask\": {
        \"display\": \"FLASK\",
        \"result\": RESULTS_DIR / \"flask\" / \"pointwise_results.jsonl\",
        \"meta\": RESULTS_DIR / \"flask\" / \"summary.json\",
    },
    \"mt_bench\": {
        \"display\": \"MT-Bench\",
        \"result\": RESULTS_DIR / \"mt_bench\" / \"pointwise_results.jsonl\",
        \"meta\": RESULTS_DIR / \"mt_bench\" / \"summary.json\",
        \"optional\": True,
    },
    \"chatbot_arena\": {
        \"display\": \"Chatbot Arena\",
        \"result\": RESULTS_DIR / \"chatbot_arena\" / \"pointwise_results.jsonl\",
        \"meta\": RESULTS_DIR / \"chatbot_arena\" / \"summary.json\",
        \"optional\": True,
    },
}

DISPLAY_NAMES = {key: cfg[\"display\"] for key, cfg in DATASET_CONFIG.items()}


def load_records(path: Path) -> List[dict]:
    with path.open(\"r\", encoding=\"utf-8\") as handle:
        return [json.loads(line) for line in handle if line.strip()]
"""

summary_fn_source = """def build_dataset_summary(dataset_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    result_path = cfg.get(\"result\")
    meta_path = cfg.get(\"meta\")
    if result_path is None or not result_path.exists():
        raise FileNotFoundError(
            f\"Could not locate result file for '{dataset_key}' at {result_path}.\"
        )

    meta_payload: Dict[str, Any] = {}
    if meta_path is not None and meta_path.exists():
        meta_payload = json.loads(meta_path.read_text(encoding=\"utf-8\"))

    analysis_sections = analyze_pointwise_results(
        result_path,
        meta_path=meta_path,
        include_meta=True,
    )

    records = load_records(result_path)
    frame = records_to_frame(records, dataset_key)

    model_scores = frame[\"model_score\"].dropna()
    human_scores = frame[\"human_mean\"].dropna()
    max_scores = frame[\"max_score\"].dropna()

    stats = {
        \"dataset\": DISPLAY_NAMES.get(dataset_key, dataset_key),
        \"examples\": int(len(frame)),
        \"with_human\": int(frame[\"human_mean\"].notna().sum()),
        \"model_mean\": float(model_scores.mean()) if not model_scores.empty else math.nan,
        \"model_std\": float(model_scores.std()) if not model_scores.empty else math.nan,
        \"human_mean\": float(human_scores.mean()) if not human_scores.empty else math.nan,
        \"max_score\": float(max_scores.max()) if not max_scores.empty else math.nan,
        \"pearson_corr\": meta_payload.get(\"pearson_corr\"),
        \"result_path\": result_path,
        \"meta_path\": meta_path,
    }

    diff_scores = frame[\"score_diff\"].dropna()
    diff_std = float(diff_scores.std()) if not diff_scores.empty else 0.0
    diff_mean = float(diff_scores.mean()) if not diff_scores.empty else 0.0
    if diff_scores.empty:
        outliers = frame.iloc[0:0]
    else:
        threshold = diff_mean + 2 * diff_std
        outliers = frame[frame[\"abs_diff\"] > threshold]

    top_examples = frame.sort_values(\"model_score\", ascending=False).head(5)
    bottom_examples = frame.sort_values(\"model_score\", ascending=True).head(5)
    largest_diffs = frame.sort_values(\"abs_diff\", ascending=False).head(5)

    summary = {
        \"key\": dataset_key,
        \"display_name\": DISPLAY_NAMES.get(dataset_key, dataset_key),
        \"config\": cfg,
        \"result_path\": result_path,
        \"meta_path\": meta_path,
        \"records\": records,
        \"meta\": meta_payload,
        \"analysis\": analysis_sections,
        \"analysis_report\": render_analysis(analysis_sections),
        \"dataframe\": frame,
        \"stats\": stats,
        \"top_examples\": top_examples[[\"id\", \"model_score\", \"human_mean\", \"score_diff\"]],
        \"bottom_examples\": bottom_examples[[\"id\", \"model_score\", \"human_mean\", \"score_diff\"]],
        \"largest_diffs\": largest_diffs[[\"id\", \"model_score\", \"human_mean\", \"score_diff\"]],
        \"outliers\": outliers[[\"id\", \"model_score\", \"human_mean\", \"score_diff\"]],
    }
    return summary
"""

summary_loop_source = """DATASET_SUMMARIES: Dict[str, Dict[str, Any]] = {}
for key, cfg in DATASET_CONFIG.items():
    if cfg.get(\"optional\") and not cfg[\"result\"].exists():
        display(Markdown(f\":information_source: Skipping optional dataset `{cfg['display']}` – no results found."))
        continue
    try:
        DATASET_SUMMARIES[key] = build_dataset_summary(key, cfg)
    except FileNotFoundError as exc:
        display(Markdown(f\":warning: {exc}\"))

AVAILABLE_DATASETS = list(DATASET_SUMMARIES.keys())
if not AVAILABLE_DATASETS:
    raise RuntimeError(\"No datasets were loaded. Generate evaluation results before running this notebook.\")
"""

# replace cells by index assumptions
code_cells_indices = [i for i, c in enumerate(nb['cells']) if c.get('cell_type') == 'code']
if len(code_cells_indices) >= 1:
    nb['cells'][code_cells_indices[0]]['source'] = [line + '\n' for line in config_source.strip().split('\n')]
if len(code_cells_indices) >= 3:
    nb['cells'][code_cells_indices[2]]['source'] = [line + '\n' for line in summary_fn_source.strip().split('\n')]
if len(code_cells_indices) >= 4:
    nb['cells'][code_cells_indices[3]]['source'] = [line + '\n' for line in summary_loop_source.strip().split('\n')]

nb_path.write_text(json.dumps(nb))
